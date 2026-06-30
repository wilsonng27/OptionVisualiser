from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from scipy.stats import norm

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})
YAHOO_OPTIONS_URL = "https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
CBOE_OPTIONS_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"


def safe_number(value, default=0.0):
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return float(value)


def fetch_chain_from_yahoo_endpoint(ticker_symbol, requested_expiration=None):
    base_response = session.get(
        YAHOO_OPTIONS_URL.format(symbol=ticker_symbol),
        timeout=8,
    )
    base_response.raise_for_status()
    base_payload = base_response.json()
    result = (base_payload.get("optionChain", {}).get("result") or [None])[0]
    if not result:
        raise ValueError("No options data returned from Yahoo endpoint.")

    expiration_timestamps = result.get("expirationDates") or []
    if not expiration_timestamps:
        raise ValueError("No expiration dates returned from Yahoo endpoint.")

    expiration_map = {
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"): ts
        for ts in expiration_timestamps
    }
    expirations = sorted(expiration_map.keys())
    selected_expiration = requested_expiration if requested_expiration in expiration_map else expirations[0]
    selected_timestamp = expiration_map[selected_expiration]

    if selected_expiration == expirations[0]:
        selected_result = result
    else:
        dated_response = session.get(
            YAHOO_OPTIONS_URL.format(symbol=ticker_symbol),
            params={"date": selected_timestamp},
            timeout=8,
        )
        dated_response.raise_for_status()
        dated_payload = dated_response.json()
        selected_result = (dated_payload.get("optionChain", {}).get("result") or [None])[0]
        if not selected_result:
            raise ValueError("No dated options data returned from Yahoo endpoint.")

    options_block = (selected_result.get("options") or [None])[0]
    if not options_block:
        raise ValueError("No options chain returned from Yahoo endpoint.")

    current_price = (
        safe_number(selected_result.get("quote", {}).get("regularMarketPrice"))
        or safe_number(selected_result.get("quote", {}).get("postMarketPrice"))
        or safe_number(selected_result.get("quote", {}).get("preMarketPrice"))
        or safe_number(selected_result.get("quote", {}).get("regularMarketPreviousClose"))
    )
    if not current_price:
        raise ValueError("Unable to determine current price from Yahoo endpoint.")

    calls = pd.DataFrame(options_block.get("calls") or [])
    puts = pd.DataFrame(options_block.get("puts") or [])
    if calls.empty or puts.empty:
        raise ValueError("Yahoo endpoint returned an empty call or put chain.")

    return current_price, expirations, selected_expiration, calls, puts


def parse_cboe_option_symbol(option_symbol):
    root = option_symbol[:-15]
    expiry_raw = option_symbol[-15:-9]
    option_type = option_symbol[-9]
    strike_raw = option_symbol[-8:]
    expiration = datetime.strptime(expiry_raw, "%y%m%d").strftime("%Y-%m-%d")
    strike = int(strike_raw) / 1000
    return root, expiration, option_type, strike


def fetch_chain_from_cboe(ticker_symbol, requested_expiration=None):
    response = session.get(
        CBOE_OPTIONS_URL.format(symbol=ticker_symbol),
        timeout=8,
    )
    response.raise_for_status()
    payload = response.json()
    data_block = payload.get("data") or {}
    rows = data_block.get("options") or []
    if not rows:
        raise ValueError("No options rows returned from CBOE.")

    current_price = safe_number(data_block.get("current_price")) or safe_number(data_block.get("close"))
    if not current_price:
        raise ValueError("Unable to determine current price from CBOE.")

    parsed_rows = []
    expiration_set = set()
    for row in rows:
        option_symbol = row.get("option")
        if not option_symbol:
            continue
        _, expiration, option_type, strike = parse_cboe_option_symbol(option_symbol)
        expiration_set.add(expiration)
        parsed_rows.append({
            "expiration": expiration,
            "option_type": option_type,
            "strike": float(strike),
            "impliedVolatility": safe_number(row.get("iv")),
            "openInterest": int(safe_number(row.get("open_interest"), 0)),
            "volume": int(safe_number(row.get("volume"), 0)),
            "lastPrice": safe_number(row.get("last_trade_price")),
            "gamma": safe_number(row.get("gamma")),
        })

    expirations = sorted(expiration_set)
    if not expirations:
        raise ValueError("No expirations parsed from CBOE options feed.")

    selected_expiration = requested_expiration if requested_expiration in expirations else expirations[0]
    selected_rows = [row for row in parsed_rows if row["expiration"] == selected_expiration]
    if not selected_rows:
        raise ValueError("No rows matched the selected CBOE expiration.")

    calls = pd.DataFrame([row for row in selected_rows if row["option_type"] == "C"])
    puts = pd.DataFrame([row for row in selected_rows if row["option_type"] == "P"])
    if calls.empty or puts.empty:
        raise ValueError("Selected CBOE expiration has empty call or put rows.")

    return current_price, expirations, selected_expiration, calls, puts


def build_liquidity_filtered_reference(data, spot):
    if not data or not spot:
        return []

    near_spot = [row for row in data if (0.88 * spot) <= row["strike"] <= (1.12 * spot)]
    if not near_spot:
        return []

    liquid_near_spot = [row for row in near_spot if row["total_oi"] > 0 or row["total_volume"] > 0]
    if not liquid_near_spot:
        return []

    avg_gross_gex = sum(row["abs_gross_gex"] for row in liquid_near_spot) / len(liquid_near_spot)
    avg_total_oi = sum(row["total_oi"] for row in liquid_near_spot) / len(liquid_near_spot)
    avg_total_volume = sum(row["total_volume"] for row in liquid_near_spot) / len(liquid_near_spot)

    filtered = [
        row for row in liquid_near_spot
        if row["abs_gross_gex"] >= (avg_gross_gex * 0.35)
        and (
            row["total_oi"] >= max(100, avg_total_oi * 0.3)
            or row["total_volume"] >= max(10, avg_total_volume * 0.3)
        )
    ]

    return filtered if filtered else liquid_near_spot


def build_level_payload(row, side):
    if not row:
        return None

    return {
        "strike": float(row["strike"]),
        "gex": float(row[f"{side}_gex"]),
        "call_oi": int(row["call_oi"]),
        "put_oi": int(row["put_oi"]),
        "total_oi": int(row["total_oi"]),
        "call_vol": int(row["call_vol"]),
        "put_vol": int(row["put_vol"]),
        "total_volume": int(row["total_volume"]),
    }

def calculate_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 0, 0, 0, 0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) * 0.01

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return delta, gamma, theta, vega


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True})


@app.route('/', methods=['GET'])
def home():
    return send_file('index.html')

@app.route('/api/flow', methods=['GET'])
def get_flow():
    ticker_symbol = request.args.get('ticker', 'SPY').upper()
    requested_exp = request.args.get('expiration', None)
    started_at = time.perf_counter()

    try:
        print(f"[flow] start ticker={ticker_symbol} requested_exp={requested_exp}", flush=True)
        try:
            source_started = time.perf_counter()
            current_price, expirations, exp_date, calls, puts = fetch_chain_from_cboe(
                ticker_symbol,
                requested_exp,
            )
            print(
                f"[flow] source=cboe fetched in {time.perf_counter() - source_started:.2f}s "
                f"exp={exp_date} calls={len(calls)} puts={len(puts)} spot={current_price}",
                flush=True,
            )
        except Exception:
            print("[flow] cboe failed, falling back to yahoo endpoint", flush=True)
            source_started = time.perf_counter()
            current_price, expirations, exp_date, calls, puts = fetch_chain_from_yahoo_endpoint(
                ticker_symbol,
                requested_exp,
            )
            print(
                f"[flow] source=yahoo fetched in {time.perf_counter() - source_started:.2f}s "
                f"exp={exp_date} calls={len(calls)} puts={len(puts)} spot={current_price}",
                flush=True,
            )

        expiry_dt = datetime.strptime(exp_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        days_to_exp = max((expiry_dt - now_utc).total_seconds() / 86400, 0)
        T = max(days_to_exp / 365.0, 0.001)
        r = 0.05
        print(f"[flow] greeks setup T={T:.6f} r={r}", flush=True)

        data = []

        # Avoid repeated DataFrame filtering inside the loop; it becomes very slow
        # on large chains and can make the API appear hung in the browser.
        map_started = time.perf_counter()
        call_map = {
            float(row["strike"]): row
            for row in calls.to_dict("records")
            if row.get("strike") is not None
        }
        put_map = {
            float(row["strike"]): row
            for row in puts.to_dict("records")
            if row.get("strike") is not None
        }
        common_strikes = sorted(set(call_map.keys()).intersection(set(put_map.keys())))
        print(
            f"[flow] maps built in {time.perf_counter() - map_started:.2f}s "
            f"common_strikes={len(common_strikes)}",
            flush=True,
        )

        loop_started = time.perf_counter()
        for strike in common_strikes:
            call = call_map[strike]
            put = put_map[strike]

            c_iv = safe_number(call.get('impliedVolatility'))
            p_iv = safe_number(put.get('impliedVolatility'))
            c_oi = int(safe_number(call.get('openInterest'), 0))
            p_oi = int(safe_number(put.get('openInterest'), 0))
            c_last = safe_number(call.get('lastPrice'))
            p_last = safe_number(put.get('lastPrice'))
            c_volume = int(safe_number(call.get('volume'), 0))
            p_volume = int(safe_number(put.get('volume'), 0))
            c_gamma_feed = safe_number(call.get('gamma'))
            p_gamma_feed = safe_number(put.get('gamma'))

            if (c_oi + p_oi) == 0 and (c_volume + p_volume) == 0:
                continue

            c_delta, c_gamma, c_theta, vega = calculate_greeks(current_price, strike, T, r, c_iv, 'call')
            p_delta, p_gamma, p_theta, _ = calculate_greeks(current_price, strike, T, r, p_iv, 'put')
            c_gamma = c_gamma_feed if c_gamma_feed > 0 else c_gamma
            p_gamma = p_gamma_feed if p_gamma_feed > 0 else p_gamma

            implied_forward = strike + (c_last - p_last) * np.exp(r * T)

            # Keep the original project GEX convention because that is the version
            # the user validated visually against their earlier deployment.
            c_gex_val = float((c_gamma * c_oi * 100 * current_price) / 1_000_000)
            p_gex_val = float((-p_gamma * p_oi * 100 * current_price) / 1_000_000)
            total_gex_val = float(c_gex_val + p_gex_val)

            data.append({
                "strike": float(strike),
                "call_vol": c_volume,
                "put_vol": p_volume,
                "call_oi": c_oi,
                "put_oi": p_oi,
                "total_oi": c_oi + p_oi,
                "total_volume": c_volume + p_volume,
                "gamma": float(round((c_gamma + p_gamma) / 2, 4)),
                "call_gex": float(round(c_gex_val, 2)),
                "put_gex": float(round(p_gex_val, 2)),
                "total_gex": float(round(total_gex_val, 2)),
                "abs_gross_gex": float(abs(c_gex_val) + abs(p_gex_val)),
                "implied_vol": float(round((c_iv + p_iv) / 2, 4)),
                "implied_forward": float(round(implied_forward, 2))
            })
        print(
            f"[flow] strike loop completed in {time.perf_counter() - loop_started:.2f}s "
            f"rows={len(data)}",
            flush=True,
        )

        net_gex = sum(d['total_gex'] for d in data)
        call_wall_row = max(data, key=lambda x: x['call_gex']) if data else None
        put_wall_row = min(data, key=lambda x: x['put_gex']) if data else None
        call_wall = call_wall_row['strike'] if call_wall_row else 0
        put_wall = put_wall_row['strike'] if put_wall_row else 0

        valid_strikes = [d for d in data if (0.90 * current_price) <= d['strike'] <= (1.10 * current_price)]
        if valid_strikes:
            avg_gross_gex = sum(d['abs_gross_gex'] for d in valid_strikes) / len(valid_strikes)
            institutional_strikes = [d for d in valid_strikes if d['abs_gross_gex'] > (avg_gross_gex * 0.5)]

            if institutional_strikes:
                zero_gamma = min(institutional_strikes, key=lambda x: abs(x['total_gex']))['strike']
            else:
                zero_gamma = min(valid_strikes, key=lambda x: abs(x['total_gex']))['strike']
        else:
            zero_gamma = 0
        print(
            f"[flow] summary net_gex={net_gex:.2f} call_wall={call_wall} put_wall={put_wall} "
            f"zero_gamma={zero_gamma} total_time={time.perf_counter() - started_at:.2f}s",
            flush=True,
        )

        return jsonify({
            "ticker": ticker_symbol,
            "spot": round(current_price, 2),
            "expiration": exp_date,
            "available_expirations": expirations,
            "net_gex": round(net_gex, 2),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "zero_gamma": zero_gamma,
            "call_wall_details": build_level_payload(call_wall_row, "call"),
            "put_wall_details": build_level_payload(put_wall_row, "put"),
            "data": data
        })

    except Exception as e:
        print(f"[flow] error after {time.perf_counter() - started_at:.2f}s: {e}", flush=True)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)



