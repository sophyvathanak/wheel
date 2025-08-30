import os, time, hmac, base64, hashlib, uuid, random
from datetime import datetime, timezone
from typing import List, Dict

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg
from psycopg.rows import dict_row

# ---- Configuration ----
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgres://user:pass@host/db
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL env var is required (Neon/Supabase/etc.)")
VOUCHER_SECRET = os.getenv("VOUCHER_SECRET", "9BA5BE5D9CEBCEF9E6AB2399D3EB2").encode()
MERCHANT_API_KEY = os.getenv("MERCHANT_API_KEY", "3B9E3")

# ---- App ----
app = FastAPI(title="Lucky Wheel API (Vercel/Serverless + Postgres)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- DB helpers ----

def db():
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)

DEFAULT_PRIZES = [
    {"name": "$5 Voucher", "quantity": 10, "win_rate": "12.5%"},
    {"name": "Free Coffee", "quantity": 10, "win_rate": "10%"},
    {"name": "Sticker Pack", "quantity": 10, "win_rate": "12.5%"},
    {"name": "10% Off", "quantity": 10, "win_rate": "12.5%"},
    {"name": "Mystery Box", "quantity": 10, "win_rate": "12.5%"},
    {"name": "T-Shirt", "quantity": 10, "win_rate": "15%"},
    {"name": "Hat", "quantity": 10, "win_rate": "10%"},
    {"name": "Grand Prize", "quantity": 5, "win_rate": "5%"},
]

DDL = (
    """
    CREATE TABLE IF NOT EXISTS prizes (
      id SERIAL PRIMARY KEY,
      name TEXT UNIQUE NOT NULL,
      quantity INTEGER NOT NULL,
      win_rate DOUBLE PRECISION NOT NULL DEFAULT 0
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS vouchers (
      voucher_id TEXT PRIMARY KEY,
      code_hash TEXT UNIQUE NOT NULL,
      prize_id INTEGER,
      prize_name TEXT,
      status TEXT NOT NULL,
      expires_at BIGINT NOT NULL,
      issued_at BIGINT NOT NULL,
      redeemed_at BIGINT,
      user_id TEXT,
      allowed_merchants TEXT
    );
    """,
)


def parse_rate_to_weight(val) -> float:
    s = str(val).strip()
    try:
        if s.endswith("%"):
            return max(0.0, float(s[:-1]))
        f = float(s)
        return max(0.0, f * 100.0 if f <= 1 else f)
    except Exception:
        return 0.0


def init_db():
    with db() as conn:
        with conn.cursor() as cur:
            for stmt in DDL:
                cur.execute(stmt)
            cur.execute("SELECT COUNT(*) AS c FROM prizes")
            if cur.fetchone()["c"] == 0:
                for p in DEFAULT_PRIZES:
                    cur.execute(
                        "INSERT INTO prizes(name, quantity, win_rate) VALUES(%s,%s,%s) ON CONFLICT (name) DO NOTHING",
                        (p["name"], p["quantity"], parse_rate_to_weight(p["win_rate"]))
                    )


init_db()  # executed on cold start of the function

# ---- Vouchers (HMAC) ----

def b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def sign(payload: str) -> str:
    return b64u(hmac.new(VOUCHER_SECRET, payload.encode(), hashlib.sha256).digest())


def new_voucher(prize_id: int, prize_name: str, ttl_min: int = 60*24*7):
    vid = uuid.uuid4().hex
    exp = int(time.time() + ttl_min*60)
    payload = f"{vid}.{prize_id}.{exp}"
    sig = sign(payload)[:24]
    code_plain = f"{vid[:8]}-{sig[:12]}".upper()
    code_hash = hashlib.sha256(code_plain.encode()).hexdigest()
    with db() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO vouchers(voucher_id, code_hash, prize_id, prize_name, status, expires_at, issued_at, allowed_merchants) 
             VALUES (%s,%s,%s,%s,'issued',%s,%s,'*')",
            (vid, code_hash, prize_id, prize_name, exp, int(datetime.now(timezone.utc).timestamp()))
        )
    return {
        "voucher_id": vid,
        "exp": exp,
        "sig": sig,
        "code": code_plain,
        "code_hash": code_hash,
        "qr_query": {"v": vid, "p": str(prize_id), "e": str(exp), "s": sig},
    }


def verify_qr(voucher_id: str, prize_id: int, exp: int, sig: str) -> bool:
    if time.time() > exp:
        return False
    payload = f"{voucher_id}.{prize_id}.{exp}"
    return hmac.compare_digest(sign(payload)[:24], sig)


# ---- Schemas ----
class ClaimBody(BaseModel):
    prize_index: int
    prize_name: str
    user_id: str | None = None


# ---- Routes ----
@app.get("/gifts")
def gifts():
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name, quantity, win_rate FROM prizes ORDER BY id")
        rows = cur.fetchall()
    # match old shape for frontend
    return {"gifts": [{"name": r["name"], "quantity": r["quantity"], "win_rate": r["win_rate"]} for r in rows]}


@app.get("/spin")
def spin():
    # Fetch ordered prize list (stable order by id)
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name, quantity, win_rate FROM prizes ORDER BY id")
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(400, "No prizes configured")

    # build availability & weights
    avail = [r for r in rows if r["quantity"] > 0 and r["win_rate"] > 0]
    if not avail:
        avail = [r for r in rows if r["quantity"] > 0]
    if not avail:
        raise HTTPException(400, "No prizes available (sold out)")

    weights = [r["win_rate"] for r in avail]
    choice = random.choices(avail, weights=weights if any(w > 0 for w in weights) else None, k=1)[0]

    # atomic decrement (retry a couple times on race)
    remaining = None
    for _ in range(3):
        with db() as conn, conn.cursor() as cur:
            cur.execute("UPDATE prizes SET quantity = quantity - 1 WHERE id = %s AND quantity > 0 RETURNING quantity", (choice["id"],))
            row = cur.fetchone()
            if row:
                remaining = row["quantity"]
                break
    if remaining is None:
        raise HTTPException(409, "Please try again")

    # compute index relative to the full ordered list
    index = next(i for i, r in enumerate(rows) if r["id"] == choice["id"])
    return {"index": index, "prize": choice["name"], "remaining": remaining}


@app.post("/claim")
def claim(body: ClaimBody, request: Request):
    # prize_index is for UI only; we store prize_name + a generic prize_id = index
    tok = new_voucher(prize_id=body.prize_index, prize_name=body.prize_name)
    base = f"{request.url.scheme}://{request.headers.get('host')}"
    qr_url = f"{base}/api/v/qr?v={tok['voucher_id']}&p={body.prize_index}&e={tok['exp']}&s={tok['sig']}"
    return {"code": tok["code"], "qr_url": qr_url, "expires_at": tok["exp"], "voucher_id": tok["voucher_id"]}


@app.get("/v/qr")
def scan_qr(v: str, p: int, e: int, s: str):
    if not verify_qr(v, p, e, s):
        raise HTTPException(400, "Invalid or expired")
    return {"voucher_id": v, "prize_index": p, "expires_at": e, "status": "scannable"}


# Merchant endpoints

def require_merchant(x_api_key: str = Header(None)):
    if not x_api_key or x_api_key != MERCHANT_API_KEY:
        raise HTTPException(401, "Invalid API key")
    return "demo-merchant"


class MerchantValidateBody(BaseModel):
    code: str | None = None
    voucher_id: str | None = None


@app.post("/merchant/vouchers/validate")
def merchant_validate(payload: MerchantValidateBody, merchant_id: str = Depends(require_merchant)):
    with db() as conn, conn.cursor() as cur:
        if payload.code:
            code_hash = hashlib.sha256(payload.code.encode()).hexdigest()
            cur.execute("SELECT voucher_id, prize_id, prize_name, status, expires_at FROM vouchers WHERE code_hash=%s", (code_hash,))
        elif payload.voucher_id:
            cur.execute("SELECT voucher_id, prize_id, prize_name, status, expires_at FROM vouchers WHERE voucher_id=%s", (payload.voucher_id,))
        else:
            raise HTTPException(400, "Provide code or voucher_id")
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "Not found")
    return {"voucher_id": row["voucher_id"], "status": row["status"], "expires_at": row["expires_at"], "prize_index": row["prize_id"], "prize_name": row["prize_name"]}


class MerchantRedeemBody(BaseModel):
    voucher_id: str


@app.post("/merchant/vouchers/redeem")
def merchant_redeem(payload: MerchantRedeemBody, merchant_id: str = Depends(require_merchant)):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT status, expires_at, allowed_merchants FROM vouchers WHERE voucher_id=%s", (payload.voucher_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(400, "NOT_FOUND")
        now = int(time.time())
        if now > int(row["expires_at"]):
            raise HTTPException(400, "EXPIRED")
        if row["status"] != "issued":
            raise HTTPException(409, "ALREADY_USED")
        allow = row.get("allowed_merchants") or "*"
        if allow != "*" and "demo-merchant" not in allow.split(","):
            raise HTTPException(400, "NOT_ALLOWED")
        cur.execute("UPDATE vouchers SET status='redeemed', redeemed_at=%s WHERE voucher_id=%s AND status='issued'", (now, payload.voucher_id))
        if cur.rowcount == 0:
            raise HTTPException(409, "RACE_CONDITION")
    return {"status": "redeemed", "voucher_id": payload.voucher_id}