import time
import pandas as pd
import yfinance as yf

TICKERS_CSV = "tickers.csv"
MAX_TICKERS = 500
BATCH_SIZE = 100
SLEEP_SEC = 0.8

MIN_TURNOVER = 50_000_000
MIN_VOLUME = 200_000
MIN_CLOSE_POSITION_FOR_WATCH = 35.0


def load_tickers():
    df = pd.read_csv(TICKERS_CSV)
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    if "name" not in df.columns:
        df["name"] = df["ticker"]

    return df.head(MAX_TICKERS)


def fetch_data(ticker):
    try:
        df = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
            group_by="column",
        )

        if df is None or df.empty or len(df) < 40:
            return None

        # MultiIndex対策
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 同名列重複対策
        df = df.loc[:, ~df.columns.duplicated()].copy()

        need_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in need_cols:
            if c not in df.columns:
                return None

        df = df[need_cols].copy()
        df = df.dropna()

        if len(df) < 40:
            return None

        # 念のため全部1次元Series化
        for c in need_cols:
            if isinstance(df[c], pd.DataFrame):
                df[c] = df[c].iloc[:, 0]

        return df

    except Exception as e:
        print(f"fetch_data error: {ticker} {e}")
        return None

def calc_indicators(df):

    df = df.copy()

    # 念のため列重複を除去
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 念のため1次元化
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]
            
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma25"] = df["Close"].rolling(25).mean()

    df["ma5_slope"] = df["ma5"] - df["ma5"].shift(1)
    df["ma25_slope"] = df["ma25"] - df["ma25"].shift(1)

    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["vol_avg3"] = df["Volume"].rolling(3).mean()

    df["volume_ratio"] = df["Volume"] / df["vol_avg20"]

    recent3 = df["Volume"].rolling(3).mean()
    prev3 = recent3.shift(3)

    df["vol_accel_3"] = recent3 / prev3

    df["ma5_gap_pct"] = (df["Close"] - df["ma5"]) / df["ma5"] * 100

    df["prev_close"] = df["Close"].shift(1)

    df["prev_change_pct"] = (
        (df["Close"] - df["prev_close"]) / df["prev_close"] * 100
    )

    df["close_5ago"] = df["Close"].shift(5)

    df["change_5d_pct"] = (
        (df["Close"] - df["close_5ago"]) / df["close_5ago"] * 100
    )

    df["day_range_pct"] = (
        (df["High"] - df["Low"]) / df["Low"] * 100
    )

    hl_range = (df["High"] - df["Low"]).replace(0, pd.NA)

    df["close_position_pct"] = (
        (df["Close"] - df["Low"]) / hl_range * 100
    )

    df["recent_high_20"] = df["High"].rolling(20).max()

    df["resistance_gap_pct"] = (
        (df["recent_high_20"] - df["Close"]) / df["Close"] * 100
    )

    df["recent_high_5"] = df["High"].rolling(5).max()

    df["breakout_gap_pct_5"] = (
        (df["recent_high_5"] - df["Close"]) / df["Close"] * 100
    )

    df["near_breakout_5"] = (df["breakout_gap_pct_5"] <= 2).astype(int)

    df["turnover"] = df["Close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000

    df["event_pre_earnings_like"] = (
        (df["vol_avg3"] > df["vol_avg20"]) &
        (df["change_5d_pct"].between(5, 20))
    ).astype(int)

    df["smart_money_absorb"] = (
        (df["volume_ratio"] >= 3) &
        (df["prev_change_pct"].abs() <= 3) &
        (df["day_range_pct"] <= 6)
    ).astype(int)

    df["core_signal"] = (
        (df["volume_ratio"] >= 1.8) &
        (df["vol_accel_3"] >= 1.3) &
        (df["close_position_pct"] >= 50)
    ).astype(int)

    return df


def score_row(row):

    score = 0

    score += min(row["volume_ratio"], 10) * 2
    score += min(row["vol_accel_3"], 5) * 1.8

    if row["near_breakout_5"] == 1:
        score += 3

    if row["resistance_gap_pct"] <= 5:
        score += 2

    close_pos = row["close_position_pct"]

    if close_pos >= 80:
        score += 3
    elif close_pos >= 65:
        score += 2
    elif close_pos >= 50:
        score += 1
    elif close_pos < 25:
        score -= 2

    gap = row["ma5_gap_pct"]

    if 0 <= gap <= 12:
        score += gap * 0.5
    elif gap > 12:
        score -= (gap - 12) * 0.8

    prev = row["prev_change_pct"]

    if 0 <= prev <= 12:
        score += prev * 0.35
    elif prev > 12:
        score -= (prev - 12) * 0.8

    if row["smart_money_absorb"] == 1:
        score += 2

    if row["event_pre_earnings_like"] == 1:
        score += 1.5

    if row["turnover_million"] >= 1000:
        score += 3
    elif row["turnover_million"] >= 500:
        score += 2

    return round(score, 2)


def add_entry_priority(df):

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df["raw_rank"] = df.index + 1

    df["entry_priority_score"] = df["score"]

    df.loc[df["raw_rank"] == 1, "entry_priority_score"] -= 2
    df.loc[df["raw_rank"] == 2, "entry_priority_score"] += 2
    df.loc[df["raw_rank"] == 3, "entry_priority_score"] += 1.5
    df.loc[df["raw_rank"].between(4, 5), "entry_priority_score"] += 1

    df = df.sort_values("entry_priority_score", ascending=False)

    df["entry_rank"] = range(1, len(df) + 1)

    return df


def run():

    tickers = load_tickers()

    rows = []

    total = len(tickers)

    for i, r in tickers.iterrows():

        ticker = r["ticker"]
        name = r["name"]

        print(f"{i+1}/{total} {ticker}")

        hist = fetch_data(ticker)

        if hist is None:
            continue

        hist = calc_indicators(hist)

        latest = hist.iloc[-1]

        if latest["turnover"] < MIN_TURNOVER:
            continue

        if latest["Volume"] < MIN_VOLUME:
            continue

        rows.append({
            "ticker": ticker,
            "name": name,
            "close": latest["Close"],
            "volume": latest["Volume"],
            "turnover_million": latest["turnover_million"],
            "volume_ratio": latest["volume_ratio"],
            "vol_accel_3": latest["vol_accel_3"],
            "close_position_pct": latest["close_position_pct"],
            "near_breakout_5": latest["near_breakout_5"],
            "event_pre_earnings_like": latest["event_pre_earnings_like"],
            "core_signal": latest["core_signal"],
            "prev_change_pct": latest["prev_change_pct"],
            "change_5d_pct": latest["change_5d_pct"],
            "ma5_gap_pct": latest["ma5_gap_pct"],
            "day_range_pct": latest["day_range_pct"],
            "resistance_gap_pct": latest["resistance_gap_pct"],
            "smart_money_absorb": latest["smart_money_absorb"],
            "score": score_row(latest)
        })

        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No candidates")
        return

    df = add_entry_priority(df)

    watch_cols = [
        "raw_rank",
        "entry_priority_score",
        "entry_rank",
        "ticker",
        "name",
        "score",
        "close",
        "turnover_million",
        "volume_ratio",
        "vol_accel_3",
        "close_position_pct",
        "near_breakout_5",
        "event_pre_earnings_like",
        "core_signal",
        "prev_change_pct",
        "change_5d_pct",
        "ma5_gap_pct",
        "day_range_pct",
        "resistance_gap_pct",
        "smart_money_absorb"
    ]

    print("\n==== Morning Watchlist ====")

    print(df[watch_cols].to_string(index=False))

    df[watch_cols].to_csv(
        "morning_watchlist.csv",
        index=False,
        encoding="utf-8-sig"
    )


if __name__ == "__main__":
    run()
