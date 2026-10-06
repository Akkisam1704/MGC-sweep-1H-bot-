"""
TopstepX (ProjectX Gateway) — Liquidity Sweep Candle Bot (GitHub Actions version).

Runs ONCE per invocation (no infinite loop) -- designed to be triggered on
a schedule by a GitHub Actions workflow. State (the timestamp of the last
candle already checked) is persisted to state.json, committed back to the
repo by the workflow after each run, so the bot knows where it left off
even though each run starts on a fresh, empty GitHub-hosted machine.

Detects the same pattern as before:
  (A) SWING_LOW_SWEEP:       pierces a confirmed left-side swing low, then
                              closes back above it (reclaim).
  (B) PREV_CANDLE_LOW_SWEEP: the prior candle was bearish, this candle's
                              low dips below that prior low, then closes
                              back above it (reclaim).

Robustness: processes EVERY candle newer than the last recorded state, not
just the single most recent one. This matters because GitHub Actions'
schedule is not guaranteed to fire exactly on time -- a delayed or skipped
run should never cause a candle to be silently missed.

Credentials come from environment variables (GitHub Actions Secrets),
not Colab Secrets:
  TOPSTEPX_USERNAME, TOPSTEPX_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

BASE_URL = "https://api.topstepx.com"
MAX_BARS_PER_REQUEST = 20000
FRACTAL_N = 2

# --- Timeframe the bot watches. Change these two to switch to 30-minute: ---
TIMEFRAME_UNIT = 3          # 1HR = 3, Minute = 2
TIMEFRAME_UNIT_NUMBER = 1   # 1HR -> 1; for 30min use unit=2, unit_number=30
TIMEFRAME_LABEL = "1HR"

LOOKBACK_DAYS = 30
STATE_FILE = "state.json"


def die(msg):
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def send_telegram_alert(bot_token, chat_id, text):
    if not bot_token or not chat_id:
        print(f"⚠️  Telegram not configured — would have sent:\n{text}")
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        if not resp.ok:
            print(f"⚠️  Telegram send failed: {resp.status_code} {resp.text}")
    except Exception as ex:
        print(f"⚠️  Telegram send error: {ex}")


# --------------------------------------------------------------------------
# State persistence
# --------------------------------------------------------------------------

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"last_checked_time": None}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# --------------------------------------------------------------------------
# REST layer
# --------------------------------------------------------------------------

def authenticate(session, username, api_key):
    resp = session.post(
        f"{BASE_URL}/api/Auth/loginKey",
        headers={"accept": "text/plain", "Content-Type": "application/json"},
        json={"userName": username, "apiKey": api_key},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        die(f"Auth failed: {data}")
    token = data.get("token")
    if not token:
        die(f"No token in response: {data}")
    print("✅ Authenticated.")
    return token


def find_mgc_contract(session, token):
    headers = {"accept": "text/plain", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    resp = session.post(f"{BASE_URL}/api/Contract/search", headers=headers,
                         json={"searchText": "MGC", "live": False}, timeout=15)
    if not resp.ok:
        die(f"Contract search failed: {resp.status_code} {resp.text}")
    results = resp.json().get("contracts") or []
    if not results:
        die("No MGC contracts found.")
    match = next((c for c in results if c.get("activeContract")), results[0])
    contract_id = match.get("id")
    print(f"✅ Using contract: {match.get('description', contract_id)} → {contract_id}")
    return contract_id


def fetch_bars(session, token, contract_id):
    headers = {"accept": "text/plain", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=LOOKBACK_DAYS)
    resp = session.post(
        f"{BASE_URL}/api/History/retrieveBars",
        headers=headers,
        json={
            "contractId": contract_id, "live": False,
            "startTime": start.isoformat(), "endTime": end.isoformat(),
            "unit": TIMEFRAME_UNIT, "unitNumber": TIMEFRAME_UNIT_NUMBER,
            "limit": MAX_BARS_PER_REQUEST, "includePartialBar": False,
        },
        timeout=30,
    )
    if not resp.ok:
        die(f"Bar fetch failed ({resp.status_code}): {resp.text}")
    bars = resp.json().get("bars", [])
    if not bars:
        die("No bars returned.")
    df = pd.DataFrame(bars).rename(columns={
        "t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------
# Fractal swing detection (no-lookahead)
# --------------------------------------------------------------------------

def find_fractal_swings(df, n=FRACTAL_N):
    highs = df["high"].values
    lows = df["low"].values
    swings = []
    for i in range(n, len(df) - n):
        window_high = highs[i - n: i + n + 1]
        window_low = lows[i - n: i + n + 1]
        if highs[i] == window_high.max() and np.argmax(window_high) == n:
            swings.append({"idx": i, "type": "high", "price": highs[i],
                            "time": df["timestamp"].iloc[i], "confirm_idx": i + n,
                            "confirm_time": df["timestamp"].iloc[i + n]})
        if lows[i] == window_low.min() and np.argmin(window_low) == n:
            swings.append({"idx": i, "type": "low", "price": lows[i],
                            "time": df["timestamp"].iloc[i], "confirm_idx": i + n,
                            "confirm_time": df["timestamp"].iloc[i + n]})
    swings.sort(key=lambda s: s["idx"])
    return clean_alternating(swings)


def clean_alternating(swings):
    if not swings:
        return []
    cleaned = [swings[0]]
    for s in swings[1:]:
        last = cleaned[-1]
        if s["type"] == last["type"]:
            if s["type"] == "high" and s["price"] > last["price"]:
                cleaned[-1] = s
            elif s["type"] == "low" and s["price"] < last["price"]:
                cleaned[-1] = s
        else:
            cleaned.append(s)
    return cleaned


def check_sweep_candle(df, swings, i):
    bar = df.iloc[i]
    prev = df.iloc[i - 1]
    if not (bar["close"] > bar["open"]):
        return [], None, None

    conditions_hit = []
    swing_lows = [s for s in swings if s["type"] == "low"]
    eligible_lows = [s for s in swing_lows if s["confirm_time"] <= bar["timestamp"] and s["idx"] < i]
    swept_swing = None
    if eligible_lows:
        candidate = eligible_lows[-1]
        if bar["low"] < candidate["price"] and bar["close"] > candidate["price"]:
            conditions_hit.append("SWING_LOW_SWEEP")
            swept_swing = candidate

    prev_bearish = prev["close"] < prev["open"]
    swept_prev_low = prev_bearish and bar["low"] < prev["low"] and bar["close"] > prev["low"]
    if swept_prev_low:
        conditions_hit.append("PREV_CANDLE_LOW_SWEEP")

    return conditions_hit, (swept_swing["price"] if swept_swing else None), (prev["low"] if swept_prev_low else None)


# --------------------------------------------------------------------------
# Main (single pass)
# --------------------------------------------------------------------------

def main():
    username = os.environ.get("TOPSTEPX_USERNAME")
    api_key = os.environ.get("TOPSTEPX_API_KEY")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not username or not api_key:
        die("Missing TOPSTEPX_USERNAME / TOPSTEPX_API_KEY environment variables.")

    state = load_state()
    last_checked_time = pd.Timestamp(state["last_checked_time"]) if state["last_checked_time"] else None

    session = requests.Session()
    token = authenticate(session, username, api_key)
    contract_id = find_mgc_contract(session, token)
    df = fetch_bars(session, token, contract_id)

    if len(df) < (2 * FRACTAL_N + 2):
        print("Not enough bars yet, skipping this run.")
        return

    swings = find_fractal_swings(df, n=FRACTAL_N)

    if last_checked_time is None:
        # First run ever: set baseline to the latest bar, don't alert
        # retroactively on candles that formed before the bot existed.
        state["last_checked_time"] = df["timestamp"].iloc[-1].isoformat()
        save_state(state)
        print(f"Baseline set: latest closed {TIMEFRAME_LABEL} candle is {df['timestamp'].iloc[-1]}. "
              f"Watching for the NEXT one onward.")
        return

    # Process EVERY bar newer than last_checked_time, in order -- not just
    # the single most recent one, so a delayed or skipped scheduled run
    # never causes a candle to be silently missed.
    new_bars = df[df["timestamp"] > last_checked_time]

    if new_bars.empty:
        print(f"No new {TIMEFRAME_LABEL} candle since {last_checked_time}. Nothing to do.")
        return

    alerts_sent = 0
    for idx in new_bars.index:
        bar = df.iloc[idx]
        conditions, swept_swing_price, prev_low = check_sweep_candle(df, swings, idx)

        if conditions:
            tag = "⭐ BOTH" if len(conditions) == 2 else conditions[0]
            extra = []
            if swept_swing_price is not None:
                extra.append(f"swept swing low {swept_swing_price:.1f}")
            if prev_low is not None:
                extra.append(f"swept prior candle low {prev_low:.1f}")
            msg = (f"🎯 LIQUIDITY SWEEP [{tag}] on {TIMEFRAME_LABEL} @ {bar['timestamp']}\n"
                   f"O={bar['open']:.1f} H={bar['high']:.1f} L={bar['low']:.1f} C={bar['close']:.1f}\n"
                   f"({'; '.join(extra)})")
            print(msg)
            send_telegram_alert(bot_token, chat_id, msg)
            alerts_sent += 1
        else:
            print(f"{bar['timestamp']}  new candle closed, no sweep pattern.")

    state["last_checked_time"] = df["timestamp"].iloc[-1].isoformat()
    save_state(state)
    print(f"Processed {len(new_bars)} new candle(s), sent {alerts_sent} alert(s). "
          f"State updated to {state['last_checked_time']}.")


if __name__ == "__main__":
    main()
