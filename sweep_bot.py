"""
TopstepX (ProjectX Gateway) — Liquidity Sweep Candle Bot (GitHub Actions version).

Runs ONCE per invocation (no infinite loop) -- designed to be triggered on
a schedule by a GitHub Actions workflow. State (the timestamp of the last
candle already checked) is persisted to state.json, committed back to the
repo by the workflow after each run, so the bot knows where it left off
even though each run starts on a fresh, empty GitHub-hosted machine.

Detects the pattern in BOTH directions:

  BULLISH (candle closes above its open):
    (A) SWING_LOW_SWEEP:       pierces a confirmed left-side swing low, then
                                closes back above it (reclaim).
    (B) PREV_CANDLE_LOW_SWEEP: the prior candle was bearish, this candle's
                                low dips below that prior low, then closes
                                back above it (reclaim).

  BEARISH (candle closes below its open) -- exact mirror:
    (A) SWING_HIGH_SWEEP:       pierces a confirmed left-side swing high,
                                 then closes back below it (reclaim).
    (B) PREV_CANDLE_HIGH_SWEEP: the prior candle was bullish, this candle's
                                 high pokes above that prior high, then
                                 closes back below it (reclaim).

Every alert is explicitly labeled BULLISH or BEARISH so you always know
which direction the signal is pointing.

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


def timeframe_delta():
    """Length of one candle, derived from the timeframe settings at the top."""
    if TIMEFRAME_UNIT == 2:
        return pd.Timedelta(minutes=TIMEFRAME_UNIT_NUMBER)
    if TIMEFRAME_UNIT == 3:
        return pd.Timedelta(hours=TIMEFRAME_UNIT_NUMBER)
    die(f"Unsupported TIMEFRAME_UNIT {TIMEFRAME_UNIT}: use 2 (minutes) or 3 (hours).")


def drop_unclosed_bars(df, now=None):
    """
    Safety net: keep only candles that have FULLY CLOSED.

    The API is already asked for closed candles only (includePartialBar=False),
    but the bot must never act on a candle that is still forming, so this does
    not rely on that flag. Bar timestamps are candle OPEN times, so a candle is
    closed once (open time + candle length) is in the past.
    """
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    closed = df[df["timestamp"] + timeframe_delta() <= now]
    dropped = len(df) - len(closed)
    if dropped:
        print(f"Ignored {dropped} unfinished candle(s) still forming.")
    return closed.reset_index(drop=True)


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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return drop_unclosed_bars(df)


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
    """
    Checks bar i for a liquidity-sweep pattern in EITHER direction:

    BULLISH (candle closes > opens):
      (A) SWING_LOW_SWEEP:       pierces a confirmed swing low, closes back above it.
      (B) PREV_CANDLE_LOW_SWEEP: prior candle was bearish, this candle's low
                                  dips below that prior low, closes back above it.

    BEARISH (candle closes < opens) -- exact mirror:
      (A) SWING_HIGH_SWEEP:       pierces a confirmed swing high, closes back below it.
      (B) PREV_CANDLE_HIGH_SWEEP: prior candle was bullish, this candle's high
                                   pokes above that prior high, closes back below it.

    A doji (close == open) matches neither direction and is skipped.
    Returns (direction, conditions_hit, swing_ref_price, prev_ref_price).
    """
    bar = df.iloc[i]
    prev = df.iloc[i - 1]

    if bar["close"] > bar["open"]:
        direction = "BULLISH"
    elif bar["close"] < bar["open"]:
        direction = "BEARISH"
    else:
        return None, [], None, None

    conditions_hit = []
    swing_ref_price = None
    prev_ref_price = None

    if direction == "BULLISH":
        swing_lows = [s for s in swings if s["type"] == "low"]
        eligible = [s for s in swing_lows if s["confirm_time"] <= bar["timestamp"] and s["idx"] < i]
        if eligible:
            candidate = eligible[-1]
            if bar["low"] < candidate["price"] and bar["close"] > candidate["price"]:
                conditions_hit.append("SWING_LOW_SWEEP")
                swing_ref_price = candidate["price"]
        prev_bearish = prev["close"] < prev["open"]
        if prev_bearish and bar["low"] < prev["low"] and bar["close"] > prev["low"]:
            conditions_hit.append("PREV_CANDLE_LOW_SWEEP")
            prev_ref_price = prev["low"]
    else:  # BEARISH
        swing_highs = [s for s in swings if s["type"] == "high"]
        eligible = [s for s in swing_highs if s["confirm_time"] <= bar["timestamp"] and s["idx"] < i]
        if eligible:
            candidate = eligible[-1]
            if bar["high"] > candidate["price"] and bar["close"] < candidate["price"]:
                conditions_hit.append("SWING_HIGH_SWEEP")
                swing_ref_price = candidate["price"]
        prev_bullish = prev["close"] > prev["open"]
        if prev_bullish and bar["high"] > prev["high"] and bar["close"] < prev["high"]:
            conditions_hit.append("PREV_CANDLE_HIGH_SWEEP")
            prev_ref_price = prev["high"]

    return direction, conditions_hit, swing_ref_price, prev_ref_price


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
        direction, conditions, swing_ref_price, prev_ref_price = check_sweep_candle(df, swings, idx)

        if conditions:
            tag = "⭐ BOTH" if len(conditions) == 2 else conditions[0]
            icon = "🎯" if direction == "BULLISH" else "🔻"
            extra = []
            if swing_ref_price is not None:
                level_word = "swing low" if direction == "BULLISH" else "swing high"
                extra.append(f"swept {level_word} {swing_ref_price:.1f}")
            if prev_ref_price is not None:
                level_word = "prior candle low" if direction == "BULLISH" else "prior candle high"
                extra.append(f"swept {level_word} {prev_ref_price:.1f}")
            msg = (f"{icon} {direction} LIQUIDITY SWEEP [{tag}] on {TIMEFRAME_LABEL} @ {bar['timestamp']}\n"
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
