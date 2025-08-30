from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timezone
import csv
import random
import threading
import os
import time
import hmac
import base64
import hashlib
import uuid
import sqlite3
from urllib.parse import urlencode

app = FastAPI(title="Lucky Wheel API (CSV + vouchers)")

# CORS (open for demo). In production, restrict to your frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Prize storage (CSV) ======
CSV_PATH = Path(__file__).with_name("prizes.csv")
LOCK = threading.Lock()
FIELDS = ["name", "quantity", "win_rate"]

DEFAULT_ROWS = [
    {"name": "$5 Voucher", "quantity": "10", "win_rate": "12.5%"},
    {"name": "Free Coffee", "quantity": "10", "win_rate": "10%"},
    {"name": "Sticker Pack", "quantity": "10", "win_rate": "12.5%"},
    {"name": "10% Off", "quantity": "10", "win_rate": "12.5%"},
    {"name": "Mystery Box", "quantity": "10", "win_rate": "12.5%"},
    {"name": "T-Shirt", "quantity": "10", "win_rate": "15%"},
    {"name": "Hat", "quantity": "10", "win_rate": "10%"},
    {"name": "Grand Prize", "quantity": "5", "win_rate": "5%"},
]


def ensure_csv_exists():
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for row in DEFAULT_ROWS:
                w.writerow(row)


def parse_rate_to_weight(rate_value) -> float:
    """Accepts "10%", 10, or 0.10 and returns a non-negative numeric weight.
    The absolute scale doesn’t matter (weights are relative)."""
    s = str(rate_value).strip()
    try:
        if s.endswith("%"):
            return max(0.0, float(s[:-1]))
        val = float(s)
        if val <= 1.0:
            return max(0.0, val * 100.0)  # treat as fraction
        return max(0.0, val)  # treat as percent already
    except Exception:
        return 0.0


def read_prizes():
    ensure_csv_exists()
    rows = []
    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            qty = int((row.get("quantity") or 0))
            rate = (row.get("win_rate") or "0").strip()
            rows.append({"name": name, "quantity": qty, "win_rate": rate})
    return rows


def write_prizes(rows):
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({
                "name": row["name"],
                "quantity": int(row["quantity"]),
                "win_rate": row["win_rate"],
            })


# ====== Voucher storage (SQLite) ======
DB_PATH = Path(__file__).with_name("vouchers.sqlite")
SECRET = os.getenv("VOUCHER_SECRET").encode()
MERCHANT_API_KEY = os.getenv("MERCHANT_API_KEY")


def db():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)  # autocommit
    conn.execute(
        """CREATE TABLE IF NOT EXISTS vouchers (
            voucher_id TEXT PRIMARY KEY,
            code_hash TEXT UNIQUE,
            prize_index INTEGER,
            prize_name TEXT,
            status TEXT,
            expires_at INTEGER,
            issued_at INTEGER,
            redeemed_at INTEGER,
            user_id TEXT,
            allowed_merchants TEXT
        )"""
    )
    return conn


def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def sign(payload: str) -> str:
    return b64u(hmac.new(SECRET, payload.encode(), hashlib.sha256).digest())


def new_voucher_token(prize_index: int, ttl_minutes: int = 60 * 24 * 7):
    vid = uuid.uuid4().hex
    exp = int(time.time() + ttl_minutes * 60)
    payload = f"{vid}.{prize_index}.{exp}"
    sig = sign(payload)[:24]
    code_plain = f"{vid[:8]}-{sig[:12]}".upper()
    code_hash = hashlib.sha256(code_plain.encode()).hexdigest()
    return {
        "voucher_id": vid,
        "exp": exp,
        "sig": sig,
        "code": code_plain,
        "code_hash": code_hash,
        "qr_query": {"v": vid, "p": str(prize_index), "e": str(exp), "s": sig},
    }


def insert_voucher(row: dict):
    conn = db()
    conn.execute(
        """INSERT INTO vouchers
        (voucher_id, code_hash, prize_index, prize_name, status, expires_at, issued_at, user_id, allowed_merchants)
        VALUES (?, ?, ?, ?, 'issued', ?, ?, ?, ?)""",
        (
            row["voucher_id"],
            row["code_hash"],
            row["prize_index"],
            row["prize_name"],
            row["expires_at"],
            row["issued_at"],
            row.get("user_id"),
            row.get("allowed_merchants", "*"),
        ),
    )
    conn.close()


def get_voucher_by_code_hash(code_hash: str):
    conn = db()
    cur = conn.execute("SELECT voucher_id, prize_index, prize_name, status, expires_at FROM vouchers WHERE code_hash=?", (code_hash,))
    row = cur.fetchone()
    conn.close()
    return row


def get_voucher(voucher_id: str):
    conn = db()
    cur = conn.execute("SELECT voucher_id, prize_index, prize_name, status, expires_at FROM vouchers WHERE voucher_id=?", (voucher_id,))
    row = cur.fetchone()
    conn.close()
    return row


def redeem_voucher(voucher_id: str, merchant_id: str):
    conn = db()
    # verify exists and valid
    cur = conn.execute("SELECT status, expires_at, allowed_merchants FROM vouchers WHERE voucher_id=?", (voucher_id,))
    row = cur.fetchone()
    if not row:
        conn.close(); return False, "NOT_FOUND"
    status, exp, allow = row
    now = int(time.time())
    if now > int(exp):
        conn.close(); return False, "EXPIRED"
    if status != "issued":
        conn.close(); return False, "ALREADY_USED"
    if allow != "*" and (merchant_id not in (allow or "").split(",")):
        conn.close(); return False, "NOT_ALLOWED"
    # atomic redeem
    conn.execute("UPDATE vouchers SET status='redeemed', redeemed_at=? WHERE voucher_id=? AND status='issued'", (now, voucher_id))
    ok = conn.total_changes > 0
    conn.close()
    return (True, "OK") if ok else (False, "RACE_CONDITION")


def verify_qr(voucher_id: str, prize_index: int, exp: int, sig: str) -> bool:
    if time.time() > exp:
        return False
    payload = f"{voucher_id}.{prize_index}.{exp}"
    return hmac.compare_digest(sign(payload)[:24], sig)


# ====== API models ======
class ClaimBody(BaseModel):
    prize_index: int
    prize_name: str
    user_id: str | None = None


# ====== FastAPI endpoints ======
@app.on_event("startup")
def startup():
    ensure_csv_exists()
    db()  # ensure tables

@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/gifts")
def get_gifts():
    """Return prizes with quantity and rate from CSV, in file order."""
    return {"gifts": read_prizes()}


@app.get("/api/spin")
def spin():
    """Weighted random pick using win_rate, skipping sold-out (quantity<=0) or 0-rate prizes.
    Decrements quantity in the CSV and returns the index relative to the /api/gifts list."""
    with LOCK:
        prizes = read_prizes()

        available = [i for i, p in enumerate(prizes) if p["quantity"] > 0 and parse_rate_to_weight(p["win_rate"]) > 0]
        if not available:
            available = [i for i, p in enumerate(prizes) if p["quantity"] > 0]
        if not available:
            raise HTTPException(status_code=400, detail="No prizes available (sold out)")

        weights = [parse_rate_to_weight(prizes[i]["win_rate"]) for i in available]
        if all(w <= 0 for w in weights):
            chosen_full_index = random.choice(available)
        else:
            pos = random.choices(range(len(available)), weights=weights, k=1)[0]
            chosen_full_index = available[pos]

        # decrement quantity and persist immediately
        prizes[chosen_full_index]["quantity"] = max(0, int(prizes[chosen_full_index]["quantity"]) - 1)
        write_prizes(prizes)

        return {
            "index": chosen_full_index,
            "prize": prizes[chosen_full_index]["name"],
            "remaining": prizes[chosen_full_index]["quantity"],
        }


@app.post("/api/claim")
def claim(body: ClaimBody):
    """Issue a single-use voucher for the given prize index/name."""
    tok = new_voucher_token(body.prize_index)
    now = int(datetime.now(timezone.utc).timestamp())
    insert_voucher({
        "voucher_id": tok["voucher_id"],
        "code_hash": tok["code_hash"],
        "prize_index": body.prize_index,
        "prize_name": body.prize_name,
        "expires_at": tok["exp"],
        "issued_at": now,
        "user_id": body.user_id,
        "allowed_merchants": "*",
    })
    base = str(request.base_url).rstrip("/")
    prefix = os.getenv("API_PREFIX", "")  # set to "/api" if this backend is deployed behind a prefix
    qr_url = f"{base}{prefix}/v/qr?{urlencode(tok['qr_query'])}"
    return {"code": tok["code"], "qr_url": qr_url, "expires_at": tok["exp"], "voucher_id": tok["voucher_id"]}


@app.get("/v/qr")
def scan_qr(v: str, p: int, e: int, s: str):
    if not verify_qr(v, p, e, s):
        raise HTTPException(400, "Invalid or expired")
    # minimal response (merchant app should still call validate/redeem)
    return {"voucher_id": v, "prize_index": p, "expires_at": e, "status": "scannable"}


# ====== Merchant endpoints (simple API key) ======
def require_merchant(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != MERCHANT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return "demo-merchant"


class MerchantValidateBody(BaseModel):
    code: str | None = None
    voucher_id: str | None = None


@app.post("/merchant/vouchers/validate")
def merchant_validate(payload: MerchantValidateBody, merchant_id: str = Depends(require_merchant)):
    row = None
    if payload.code:
        code_hash = hashlib.sha256(payload.code.encode()).hexdigest()
        row = get_voucher_by_code_hash(code_hash)
    elif payload.voucher_id:
        row = get_voucher(payload.voucher_id)
    else:
        raise HTTPException(400, "Provide code or voucher_id")

    if not row:
        raise HTTPException(404, "Not found")
    vid, prize_idx, prize_name, status, exp = row
    return {"voucher_id": vid, "status": status, "expires_at": exp, "prize_index": prize_idx, "prize_name": prize_name}


class MerchantRedeemBody(BaseModel):
    voucher_id: str


@app.post("/merchant/vouchers/redeem")
def merchant_redeem(payload: MerchantRedeemBody, merchant_id: str = Depends(require_merchant)):
    ok, why = redeem_voucher(payload.voucher_id, merchant_id)
    if not ok:
        raise HTTPException(409 if why == "ALREADY_USED" else 400, why)
    return {"status": "redeemed", "voucher_id": payload.voucher_id}