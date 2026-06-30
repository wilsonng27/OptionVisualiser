from datetime import datetime, timezone

import numpy as np
import requests
import yfinance as yf
from flask import Flask, jsonify, request
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


def safe_number(value, default=0.0):
    if value is None:
        return default
    try:
        if np.isnan(value):
            return default
    except TypeError:
        pass
    return float(value)


def get_current_price(ticker):
    fast_info = getattr(ticker, "fast_info", {}) or {}
    for key in ("lastPrice", "regularMarketPrice", "previousClose"):
        value = fast_info.get(key)
        if value:
            return float(value)

    history = ticker.history(period="1d", interval="1m")
    if not history.empty:
        return float(history["Close"].dropna().iloc[-1])

    raise ValueError("Unable to determine current price.")


def build_liquidity_filtered_reference(data, spot):
    if not data or not spot:
        return []

    near_spot = [row for row in data if (0.88 * spot) <= row["strike"] <= (1.12 * spot)]
    if not near_spot:
        return []

    avg_gross_gex = sum(row["abs_gross_gex"] for row in near_spot) / len(near_spot)
    avg_total_oi = sum(row["total_oi"] for row in near_spot) / len(near_spot)
    avg_total_volume = sum(row["total_volume"] for row in near_spot) / len(near_spot)

    filtered = [
        row for row in near_spot
        if row["abs_gross_gex"] >= (avg_gross_gex * 0.35)
        and (
            row["total_oi"] >= max(100, avg_total_oi * 0.3)
            or row["total_volume"] >= max(10, avg_total_volume * 0.3)
        )
    ]

    return filtered if filtered else near_spot


def build_peak_payload(row, side):
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

@app.route('/api/flow', methods=['GET'])
def get_flow():
    ticker_symbol = request.args.get('ticker', 'SPY').upper()
    requested_exp = request.args.get('expiration', None)
    ticker = yf.Ticker(ticker_symbol, session=session)

    try:
        current_price = get_current_price(ticker)
        expirations = list(ticker.options)

        if not expirations:
            return jsonify({"error": "No options found."})

        exp_date = requested_exp if requested_exp in expirations else expirations[0]
        chain = ticker.option_chain(exp_date)

        expiry_dt = datetime.strptime(exp_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)
        days_to_exp = max((expiry_dt - now_utc).total_seconds() / 86400, 0)
        T = max(days_to_exp / 365.0, 0.001)
        r = 0.05

        calls = chain.calls
        puts = chain.puts

        data = []
        common_strikes = set(calls['strike']).intersection(set(puts['strike']))

        for strike in sorted(common_strikes):
            call = calls[calls['strike'] == strike].iloc[0]
            put = puts[puts['strike'] == strike].iloc[0]

            c_iv = safe_number(call.get('impliedVolatility'))
            p_iv = safe_number(put.get('impliedVolatility'))
            c_oi = int(safe_number(call.get('openInterest'), 0))
            p_oi = int(safe_number(put.get('openInterest'), 0))
            c_last = safe_number(call.get('lastPrice'))
            p_last = safe_number(put.get('lastPrice'))
            c_volume = int(safe_number(call.get('volume'), 0))
            p_volume = int(safe_number(put.get('volume'), 0))

            c_delta, c_gamma, c_theta, vega = calculate_greeks(current_price, strike, T, r, c_iv, 'call')
            p_delta, p_gamma, p_theta, _ = calculate_greeks(current_price, strike, T, r, p_iv, 'put')

            implied_forward = strike + (c_last - p_last) * np.exp(r * T)

            # Express GEX as dollar gamma for a 1% underlying move, scaled to millions.
            c_gex_val = float((c_gamma * c_oi * 100 * (current_price ** 2) * 0.01) / 1_000_000)
            p_gex_val = float((-p_gamma * p_oi * 100 * (current_price ** 2) * 0.01) / 1_000_000)
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

        net_gex = sum(d['total_gex'] for d in data)
        reference_strikes = build_liquidity_filtered_reference(data, current_price)
        peak_call = None
        peak_put = None

        if reference_strikes:
            peak_call = max(reference_strikes, key=lambda x: x['call_gex'])
            peak_put = min(reference_strikes, key=lambda x: x['put_gex'])
            call_wall = peak_call['strike']
            put_wall = peak_put['strike']
            zero_gamma = min(reference_strikes, key=lambda x: abs(x['total_gex']))['strike']
        else:
            peak_call = max(data, key=lambda x: x['call_gex']) if data else None
            peak_put = min(data, key=lambda x: x['put_gex']) if data else None
            call_wall = peak_call['strike'] if peak_call else 0
            put_wall = peak_put['strike'] if peak_put else 0
            zero_gamma = 0

        return jsonify({
            "ticker": ticker_symbol,
            "spot": round(current_price, 2),
            "expiration": exp_date,
            "available_expirations": expirations,
            "net_gex": round(net_gex, 2),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "zero_gamma": zero_gamma,
            "peak_call": build_peak_payload(peak_call, "call"),
            "peak_put": build_peak_payload(peak_put, "put"),
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)

