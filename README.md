OpenAlpha Day Trader

OpenAlpha Day Trader is a local Python and Streamlit application for intraday stock and ETF analysis.
It continuously scans a configurable universe (such as S&P 500 ETFs or custom symbols) and generates ranked buy/sell signals based on probability models, technical rules, and volatility-adjusted position sizing.

Features

Live scanning of U.S. stocks and ETFs with configurable intervals (1m, 5m, or 1d)

Blended signal engine using logistic regression and rule-based scoring

Volatility-based position sizing and portfolio caps

Transparent, fully local logic (no black-box AI)

Free data through TwelveData API (API key required)

Streamlit web interface with three tabs: Live, Backtest, and Settings

Built-in backtesting with basic metrics and equity curves

Runs entirely inside Visual Studio Code on Windows

Prerequisites

Python 3.10 or newer

Git

A free TwelveData API key (https://twelvedata.com
)

Visual Studio Code (recommended for editing and running)

Installation

Clone the repository:
git clone https://github.com/BasirOskoei/openalpha_day_trader.git

cd openalpha_day_trader

Create and activate a Python virtual environment:
python -m venv .venv
..venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Verify installation:
python -c "import streamlit, pandas, requests; print('Environment ready')"

Setting the TwelveData API Key

The app requires a TwelveData key to fetch live market data.
You can set it in PowerShell before running the app:

$env:TWELVEDATA_KEY = "YOUR_API_KEY_HERE"

You can confirm the key is available to Python:

python -c "import os; print(os.getenv('TWELVEDATA_KEY'))"

If it prints your key, you’re ready.

Running the App

From inside the project folder:

python -m streamlit run ui\streamlit_app.py

After a few seconds, Streamlit will print a local URL such as:
Local URL: http://localhost:8501

Open that link in your browser to access the dashboard.

Tabs Overview

Live

Continuously scans your universe and shows:

Current top pick (symbol, action, and position size)

Confidence, probabilities, and rule scores

Top 10 ranked symbols

Live updating chart (Close, SMA20, SMA50, SMA200)

“Run scheduler” automatically refreshes on the chosen interval.

Backtest

Runs a historical simulation using daily bars.

Displays performance metrics and equity curve.

Settings

Modify scan interval, thresholds, weights, and universe.

Upload your own universe CSV (symbol header required).

Adjust risk budget and max exposure for your equity size.

Data Provider

By default, OpenAlpha uses the TwelveData API with a free key for 1–5 minute delayed data.
If no key is provided, the provider will fail to load.
To use live or premium feeds, insert your own API key in the environment or update app/data_providers.py.

Troubleshooting

Error: TWELVEDATA_KEY is not set
→ You forgot to set your API key. Use:
$env:TWELVEDATA_KEY = "YOUR_KEY"

Empty chart or data
→ Free TwelveData accounts sometimes limit requests per minute. Try switching interval from 1m to 5m in Settings.

All sizes show 0
→ Increase risk settings in the Settings tab:
Risk budget per trade: 1.0
Max name exposure: 1.0

App not refreshing automatically
→ Ensure “Run scheduler” is toggled ON in the Live tab.
