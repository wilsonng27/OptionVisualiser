from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from scipy.stats import norm

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


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
    ticker = yf.Ticker(ticker_symbol)

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

            c_delta, gamma, c_theta, vega = calculate_greeks(current_price, strike, T, r, c_iv, 'call')
            p_delta, _, p_theta, _ = calculate_greeks(current_price, strike, T, r, p_iv, 'put')

            implied_forward = strike + (c_last - p_last) * np.exp(r * T)

            c_gex = (gamma * c_oi * 100 * current_price) / 1_000_000
            p_gex = (-gamma * p_oi * 100 * current_price) / 1_000_000
            total_gex = c_gex + p_gex

            data.append({
                "strike": strike,
                "call_vol": c_volume,
                "put_vol": p_volume,
                "gamma": round(gamma, 4),
                "call_gex": round(c_gex, 2),
                "put_gex": round(p_gex, 2),
                "total_gex": round(total_gex, 2),
                "abs_gross_gex": abs(c_gex) + abs(p_gex),
                "implied_vol": round((c_iv + p_iv) / 2, 4),
                "implied_forward": round(implied_forward, 2)
            })

        net_gex = sum(d['total_gex'] for d in data)
        call_wall = max(data, key=lambda x: x['call_gex'])['strike'] if data else 0
        put_wall = min(data, key=lambda x: x['put_gex'])['strike'] if data else 0

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

        return jsonify({
            "ticker": ticker_symbol,
            "spot": round(current_price, 2),
            "expiration": exp_date,
            "available_expirations": expirations,
            "net_gex": round(net_gex, 2),
            "call_wall": call_wall,
            "put_wall": put_wall,
            "zero_gamma": zero_gamma,
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)

