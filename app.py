


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Indian Stock Screener",
    layout="wide"
)

st.title("🇮🇳 Indian Stock Screener")
st.subheader("Swing Trading Dashboard")

stocks = [
    "RELIANCE", "HDFCBANK", "ICICIBANK",
    "SBIN", "INFY", "TCS", "ITC", "LT",
    "AXISBANK", "KOTAKBANK", "FEDERALBNK",
    "BEL", "HAL", "TRENT", "ZOMATO",
    "TATAMOTORS", "BHARTIARTL", "NTPC",
    "POWERGRID", "BAJFINANCE"
]

@st.cache_data(ttl=3600)
def run_screener():
    results = []

    for symbol in stocks:
        try:
            df = yf.download(
                symbol + ".NS",
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 200:
                continue

            close = df["Close"]
            volume = df["Volume"]

            sma50 = close.rolling(50).mean()
            sma200 = close.rolling(200).mean()

            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.ewm(
                alpha=1/14, adjust=False,
                min_periods=14
            ).mean()

            avg_loss = loss.ewm(
                alpha=1/14, adjust=False,
                min_periods=14
            ).mean()

            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()

            avg_volume = volume.rolling(20).mean()
            volume_ratio = volume / avg_volume

            previous_high = df["High"].shift(1).rolling(20).max()

            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - close.shift()).abs()
            low_close = (df["Low"] - close.shift()).abs()

            tr = pd.concat(
                [high_low, high_close, low_close], axis=1
            ).max(axis=1)

            atr = tr.rolling(14).mean()

            price = float(close.iloc[-1])
            s50 = float(sma50.iloc[-1])
            s200 = float(sma200.iloc[-1])
            r = float(rsi.iloc[-1])
            vr = float(volume_ratio.iloc[-1])
            a = float(atr.iloc[-1])

            if not all(np.isfinite(x) for x in [price, s50, s200, r, vr, a]):
                continue

            score = 0
            score += 25 if price > s50 else 0
            score += 20 if s50 > s200 else 0
            score += 15 if 50 <= r <= 72 else 0
            score += 15 if macd.iloc[-1] > signal.iloc[-1] else 0
            score += 15 if vr >= 1.5 else (8 if vr >= 1 else 0)

            breakout = price > float(previous_high.iloc[-1])
            score += 10 if breakout else 0

            stop = price - 1.5 * a
            target = price + 2 * (price - stop)

            results.append({
                "Symbol": symbol,
                "Score": score,
                "Price": round(price, 2),
                "RSI": round(r, 2),
                "Volume Ratio": round(vr, 2),
                "Breakout": "Yes" if breakout else "No",
                "Stop Loss": round(stop, 2),
                "Target": round(target, 2)
            })

        except Exception:
            continue

    return pd.DataFrame(results)

if st.button("Run Stock Screener"):
    with st.spinner("Downloading NSE stock data..."):
        result = run_screener()

    if result.empty:
        st.warning("No data available. Please try again later.")
    else:
        result = result.sort_values(
            "Score", ascending=False
        ).reset_index(drop=True)

        st.metric("Stocks Scanned", len(result))
        st.metric(
            "Strong Setups (Score 75+)",
            int((result["Score"] >= 75).sum())
        )

        st.subheader("Top Swing Trading Stocks")
        st.dataframe(
            result.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Complete Stock Ranking")
        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Stock Report",
            csv,
            "Indian_Stock_Screener.csv",
            "text/csv"
        )

st.caption(
    "Educational screening tool. Prices and signals are not investment advice."
)
