
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time

st.set_page_config(
    page_title="NSE All Stock Screener",
    layout="wide"
)

st.title("🇮🇳 NSE Mainboard Stock Screener")
st.caption("Market Cap above ₹1,000 Crore | Swing Trading")

MIN_MARKET_CAP = 1000

NSE_URL = (
    "https://nsearchives.nseindia.com/"
    "content/equities/EQUITY_L.csv"
)

@st.cache_data(ttl=86400)
def get_nse_stocks():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*"
    }

    response = requests.get(
        NSE_URL,
        headers=headers,
        timeout=30
    )
    response.raise_for_status()

    df = pd.read_csv(io.BytesIO(response.content))
    df.columns = df.columns.str.strip().str.upper()

    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    df = df[df["SERIES"] == "EQ"]
    df = df[df["SYMBOL"].str.match(r"^[A-Z0-9&-]+$")]

    return sorted(df["SYMBOL"].dropna().unique().tolist())


@st.cache_data(ttl=86400)
def get_market_cap(symbol):
    try:
        ticker = yf.Ticker(symbol + ".NS")
        fast = ticker.fast_info

        cap = fast.get("market_cap")

        if cap is None:
            cap = ticker.get_info().get("marketCap")

        if cap is None:
            return None

        return float(cap) / 10000000

    except Exception:
        return None


def analyze_stock(symbol, df, market_cap):
    try:
        if df is None or df.empty or len(df) < 200:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        volume = df["Volume"]

        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1/14,
            adjust=False,
            min_periods=14
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/14,
            adjust=False,
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

        previous_high = (
            df["High"].shift(1).rolling(20).max()
        )

        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - close.shift()).abs()
        low_close = (df["Low"] - close.shift()).abs()

        tr = pd.concat(
            [high_low, high_close, low_close],
            axis=1
        ).max(axis=1)

        atr = tr.rolling(14).mean()

        price = float(close.iloc[-1])
        r = float(rsi.iloc[-1])
        s50 = float(sma50.iloc[-1])
        s200 = float(sma200.iloc[-1])
        vr = float(volume_ratio.iloc[-1])
        a = float(atr.iloc[-1])

        if not all(np.isfinite(x) for x in
                   [price, r, s50, s200, vr, a]):
            return None

        score = 0

        if price > s50:
            score += 25

        if s50 > s200:
            score += 20

        if 50 <= r <= 72:
            score += 15

        if macd.iloc[-1] > signal.iloc[-1]:
            score += 15

        if vr >= 1.5:
            score += 15
        elif vr >= 1:
            score += 8

        breakout = price > float(previous_high.iloc[-1])

        if breakout:
            score += 10

        stop = price - 1.5 * a
        target = price + 2 * (price - stop)

        return {
            "Symbol": symbol,
            "Market Cap (Cr)": round(market_cap, 2),
            "Score": score,
            "Price": round(price, 2),
            "RSI": round(r, 2),
            "SMA20": round(float(sma20.iloc[-1]), 2),
            "SMA50": round(s50, 2),
            "SMA200": round(s200, 2),
            "Volume Ratio": round(vr, 2),
            "Breakout": "Yes" if breakout else "No",
            "Stop Loss": round(stop, 2),
            "Target": round(target, 2)
        }

    except Exception:
        return None


if st.button("Scan All NSE Mainboard Stocks"):

    try:
        with st.spinner("Downloading NSE stock list..."):
            symbols = get_nse_stocks()

        st.success(f"NSE EQ symbols found: {len(symbols)}")

        qualified = []
        progress = st.progress(0)
        status = st.empty()

        for i, symbol in enumerate(symbols):
            cap = get_market_cap(symbol)

            if cap is not None and cap > MIN_MARKET_CAP:
                qualified.append((symbol, cap))

            progress.progress((i + 1) / len(symbols))
            status.write(
                f"Checking market cap: {i+1}/{len(symbols)}"
            )

        st.success(
            f"Stocks above ₹1,000 Cr: {len(qualified)}"
        )

        results = []
        symbols_to_scan = [
            symbol for symbol, cap in qualified
        ]

        cap_map = dict(qualified)

        chunks = [
            symbols_to_scan[i:i+40]
            for i in range(0, len(symbols_to_scan), 40)
        ]

        scan_progress = st.progress(0)
        scan_status = st.empty()

        for i, batch in enumerate(chunks):
            data = yf.download(
                [s + ".NS" for s in batch],
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True
            )

            for symbol in batch:
                try:
                    ticker = symbol + ".NS"

                    if isinstance(data.columns, pd.MultiIndex):
                        df = data[ticker].copy()
                    else:
                        df = data.copy()

                    result = analyze_stock(
                        symbol,
                        df,
                        cap_map[symbol]
                    )

                    if result:
                        results.append(result)

                except Exception:
                    continue

            scan_progress.progress((i + 1) / len(chunks))
            scan_status.write(
                f"Technical scanning batch {i+1}/{len(chunks)}"
            )

        if results:
            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "Score",
                ascending=False
            ).reset_index(drop=True)

            st.subheader("Screening Summary")

            c1, c2, c3 = st.columns(3)

            c1.metric("Eligible Stocks", len(qualified))
            c2.metric("Successfully Scanned", len(result_df))
            c3.metric(
                "Strong Setups",
                int((result_df["Score"] >= 75).sum())
            )

            st.subheader("Top Swing Trading Stocks")

            st.dataframe(
                result_df.head(20),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Complete Stock Ranking")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            csv = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Complete Stock Report",
                csv,
                "NSE_Mainboard_Screener.csv",
                "text/csv"
            )

        else:
            st.warning(
                "No stocks were successfully scanned. "
                "Try again later or check data availability."
            )

    except Exception as e:
        st.error(f"Scanner error: {e}")

st.caption(
    "Market cap data is sourced from Yahoo Finance and may "
    "be unavailable or delayed. This is an educational tool, "
    "not investment advice."
)
