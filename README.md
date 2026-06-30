# Options Flow & GEX Visualizer

A professional-grade quantitative dashboard for visualizing options chain liquidity, Gamma Exposure (GEX), and market structure. Built with Flask and Chart.js.

## Overview
This tool provides a structural analysis of options chains, including:
- **GEX Profiles:** Visualize Call and Put gamma exposure.
- **Liquidity Analysis:** Identify Call Walls, Put Walls, and Zero Gamma levels.
- **Synthetic Forward Curve:** Monitor pricing efficiency across the chain.

## Technologies Used
- **Backend:** Python, Flask, yfinance, SciPy (Black-Scholes model).
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js.
- **Deployment:** Render (Backend), GitHub Pages (Frontend).

## Quick Start
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/options-visualizer.git`
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the server: `python app.py`

## Warning
⚠️ **Disclaimer:** This tool uses delayed market data (15-20 min) and is intended for educational and structural analysis only. Not for 0DTE execution.

## License
MIT License - See LICENSE file for details.
