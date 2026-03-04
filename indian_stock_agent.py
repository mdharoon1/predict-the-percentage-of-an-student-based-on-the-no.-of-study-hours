"""
Indian Stock Trading Agent
===========================
Generates entry and exit signals for Indian (NSE/BSE) stocks using
technical indicators: Moving Average crossover, RSI, and MACD.

Usage:
    python indian_stock_agent.py --symbol RELIANCE.NS --period 6mo
    python indian_stock_agent.py --symbol TCS.NS --period 1y
    python indian_stock_agent.py --symbol INFY.NS --period 3mo

NSE symbols end with '.NS' (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS)
BSE symbols end with '.BO' (e.g., RELIANCE.BO)
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


# ── Technical Indicators ────────────────────────────────────────────────────

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = 12,
                 slow: int = 26,
                 signal: int = 9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(series: pd.Series,
                            window: int = 20,
                            num_std: float = 2.0):
    sma = compute_sma(series, window)
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


# ── Signal Generation ────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine MA crossover, RSI, and MACD into a unified BUY / SELL / HOLD
    signal.

    Entry (BUY)  conditions (at least 2 of 3 must be true):
        1. Short MA (20) crosses above Long MA (50)  — bullish crossover
        2. RSI(14) rises above 30 from oversold territory (<= 30)
        3. MACD line crosses above signal line

    Exit (SELL) conditions (at least 2 of 3 must be true):
        1. Short MA (20) crosses below Long MA (50)  — bearish crossover
        2. RSI(14) falls below 70 from overbought territory (>= 70)
        3. MACD line crosses below signal line
    """
    close = df["Close"].squeeze()

    df["SMA_20"] = compute_sma(close, 20)
    df["SMA_50"] = compute_sma(close, 50)
    df["RSI"] = compute_rsi(close, 14)
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = compute_macd(close)
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = compute_bollinger_bands(close)

    # MA crossover
    df["MA_Bull"] = (df["SMA_20"] > df["SMA_50"]) & (
        df["SMA_20"].shift(1) <= df["SMA_50"].shift(1)
    )
    df["MA_Bear"] = (df["SMA_20"] < df["SMA_50"]) & (
        df["SMA_20"].shift(1) >= df["SMA_50"].shift(1)
    )

    # RSI signal
    df["RSI_Bull"] = (df["RSI"] > 30) & (df["RSI"].shift(1) <= 30)
    df["RSI_Bear"] = (df["RSI"] < 70) & (df["RSI"].shift(1) >= 70)

    # MACD crossover
    df["MACD_Bull"] = (df["MACD"] > df["MACD_Signal"]) & (
        df["MACD"].shift(1) <= df["MACD_Signal"].shift(1)
    )
    df["MACD_Bear"] = (df["MACD"] < df["MACD_Signal"]) & (
        df["MACD"].shift(1) >= df["MACD_Signal"].shift(1)
    )

    bull_score = df["MA_Bull"].astype(int) + df["RSI_Bull"].astype(int) + df["MACD_Bull"].astype(int)
    bear_score = df["MA_Bear"].astype(int) + df["RSI_Bear"].astype(int) + df["MACD_Bear"].astype(int)

    df["Signal"] = "HOLD"
    df.loc[bull_score >= 2, "Signal"] = "BUY"
    df.loc[bear_score >= 2, "Signal"] = "SELL"

    return df


# ── Backtesting ──────────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame, initial_capital: float = 100_000.0) -> dict:
    """Simple long-only backtest based on BUY/SELL signals."""
    capital = initial_capital
    position = 0
    entry_price = 0.0
    trades = []

    close = df["Close"].squeeze()

    for date, row in df.iterrows():
        price = float(close.loc[date])
        signal = row["Signal"]

        if signal == "BUY" and position == 0:
            shares = int(capital // price)
            if shares > 0:
                position = shares
                entry_price = price
                capital -= shares * price
                trades.append({
                    "Date": date,
                    "Action": "BUY",
                    "Price": round(price, 2),
                    "Shares": shares,
                    "Capital": round(capital, 2),
                })

        elif signal == "SELL" and position > 0:
            capital += position * price
            pnl = (price - entry_price) * position
            trades.append({
                "Date": date,
                "Action": "SELL",
                "Price": round(price, 2),
                "Shares": position,
                "P&L": round(pnl, 2),
                "Capital": round(capital, 2),
            })
            position = 0
            entry_price = 0.0

    # Close any open position at last price
    if position > 0:
        last_price = float(close.iloc[-1])
        capital += position * last_price
        unrealised_pnl = (last_price - entry_price) * position
        trades.append({
            "Date": df.index[-1],
            "Action": "HOLD (open)",
            "Price": round(last_price, 2),
            "Shares": position,
            "P&L (unrealised)": round(unrealised_pnl, 2),
            "Capital": round(capital, 2),
        })

    total_return_pct = ((capital - initial_capital) / initial_capital) * 100
    return {
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "trades": trades,
    }


# ── Recommendation ──────────────────────────────────────────────────────────

def current_recommendation(df: pd.DataFrame, symbol: str) -> None:
    last = df.iloc[-1]
    close_price = float(df["Close"].squeeze().iloc[-1])

    print("\n" + "=" * 60)
    print(f"  Indian Stock Agent — {symbol}")
    print("=" * 60)
    print(f"  Date          : {df.index[-1].date()}")
    print(f"  Close Price   : ₹{close_price:,.2f}")
    print(f"  SMA 20        : ₹{last['SMA_20']:,.2f}")
    print(f"  SMA 50        : ₹{last['SMA_50']:,.2f}")
    print(f"  RSI (14)      : {last['RSI']:.1f}")
    print(f"  MACD          : {last['MACD']:.4f}")
    print(f"  MACD Signal   : {last['MACD_Signal']:.4f}")
    print(f"  BB Upper      : ₹{last['BB_Upper']:,.2f}")
    print(f"  BB Lower      : ₹{last['BB_Lower']:,.2f}")

    signal = last["Signal"]
    if signal == "BUY":
        print(f"\n  ✅  RECOMMENDATION : ENTRY (BUY)")
    elif signal == "SELL":
        print(f"\n  🚨  RECOMMENDATION : EXIT (SELL)")
    else:
        print(f"\n  ⏳  RECOMMENDATION : HOLD")

    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_agent(symbol: str, period: str = "6mo") -> None:
    print(f"\nFetching data for {symbol} (period={period}) …")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        print(f"ERROR: No data returned for symbol '{symbol}'.")
        print("Make sure to append '.NS' for NSE (e.g. RELIANCE.NS) "
              "or '.BO' for BSE (e.g. RELIANCE.BO).")
        sys.exit(1)

    df = generate_signals(df)

    current_recommendation(df, symbol)

    # Show recent signals
    signals = df[df["Signal"].isin(["BUY", "SELL"])][
        ["Close", "SMA_20", "SMA_50", "RSI", "MACD", "Signal"]
    ].copy()
    signals["Close"] = signals["Close"].apply(lambda x: f"₹{float(x):,.2f}")

    if not signals.empty:
        print("\nRecent BUY / SELL signals:")
        print(signals.tail(10).to_string())
    else:
        print("\nNo BUY / SELL signals generated in this period.")

    # Backtest summary
    result = backtest(df)
    print(f"\nBacktest Summary ({period}):")
    print(f"  Initial Capital : ₹{result['initial_capital']:,.2f}")
    print(f"  Final Capital   : ₹{result['final_capital']:,.2f}")
    print(f"  Total Return    : {result['total_return_pct']:.2f}%")
    print(f"  Total Trades    : {len(result['trades'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Indian Stock Entry/Exit Agent"
    )
    parser.add_argument(
        "--symbol",
        default="RELIANCE.NS",
        help="Stock symbol with exchange suffix, e.g. RELIANCE.NS or TCS.NS",
    )
    parser.add_argument(
        "--period",
        default="6mo",
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        help="Historical data period (default: 6mo)",
    )
    args = parser.parse_args()
    run_agent(args.symbol, args.period)


if __name__ == "__main__":
    main()
