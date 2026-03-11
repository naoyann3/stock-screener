import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


# ============================================================
# Japanese Stock Short-Term Momentum Screener v3
# Base: user's current screener.py
# Improvements:
#   1) previous-day volume > 20-day average flag/score
#   2) institutional accumulation detection logic
#   3) scoring overhaul with bonus / penalty balance
# ============================================================

TICKERS_CSV = "tickers.csv"
OUTPUT_CSV = "morning_watchlist_v3.csv"
MAX_TICKERS = 500
BATCH_SIZE = 100
SLEEP_SEC = 0.8

# Liquidity filters
MIN_TURNOVER = 50_000_000
MIN_VOLUME = 200_000

# Trend / momentum filters
MIN_CLOSE_POSITION_FOR_WATCH = 35.0
MAX_MA5_GAP_PCT = 14.0
MAX_PREV_CHANGE_PCT = 12.0
MAX_DAY_RANGE_PCT = 9.0

# Score / selection
TOP_N_OUTPUT = 80


@dataclass
class ScreenerConfig:
    min_turnover: int = MIN_TURNOVER
    min_volume: int = MIN_VOLUME
    min_close_position_for_watch: float = MIN_CLOSE_POSITION_FOR_WATCH
    max_ma5_gap_pct: float = MAX_MA5_GAP_PCT
    max_prev_change_pct: float = MAX_PREV_CHANGE_PCT
    max_day_range_pct: float = MAX_DAY_RANGE_PCT


CFG = ScreenerConfig()


def load_tickers() -> pd.DataFrame:
    df = pd.read_csv(TICKERS_CSV)
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()

    if "name" not in df.columns:
        df["name"] = df["ticker"]

    return df.head(MAX_TICKERS)


def fetch_data(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period="8mo",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=False,
            group_by="column",
        )

        if df is None or df.empty or len(df) < 50:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()].copy()

        need_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in need_cols:
            if c not in df.columns:
                return None

        df = df[need_cols].copy().dropna()

        if len(df) < 50:
            return None

        for c in need_cols:
            if isinstance(df[c], pd.DataFrame):
                df[c] = df[c].iloc[:, 0]

        return df

    except Exception as e:
        print(f"fetch_data error: {ticker} {e}")
        return None


def safe_div(a, b):
    return a / b.replace(0, pd.NA)


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]

    # Moving averages
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma25"] = df["Close"].rolling(25).mean()

    df["ma5_slope"] = df["ma5"] - df["ma5"].shift(1)
    df["ma10_slope"] = df["ma10"] - df["ma10"].shift(1)
    df["ma25_slope"] = df["ma25"] - df["ma25"].shift(1)

    # Volume
    df["vol_avg5"] = df["Volume"].rolling(5).mean()
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["vol_avg60"] = df["Volume"].rolling(60).mean()

    df["volume_ratio"] = safe_div(df["Volume"], df["vol_avg20"])
    df["volume_ratio_60"] = safe_div(df["Volume"], df["vol_avg60"])

    recent3 = df["Volume"].rolling(3).mean()
    prev3 = recent3.shift(3)
    df["vol_accel_3"] = safe_div(recent3, prev3)

    # Previous-day / multi-day volume context
    df["prev_volume"] = df["Volume"].shift(1)
    df["prev_vol_avg20"] = df["vol_avg20"].shift(1)
    df["prev_day_vol_gt_20d"] = (df["prev_volume"] > df["prev_vol_avg20"]).astype(int)
    df["prev_day_vol_ratio_20"] = safe_div(df["prev_volume"], df["prev_vol_avg20"])

    # Price / momentum
    df["prev_close"] = df["Close"].shift(1)
    df["prev_change_pct"] = safe_div(df["Close"] - df["prev_close"], df["prev_close"]) * 100

    df["close_3ago"] = df["Close"].shift(3)
    df["close_5ago"] = df["Close"].shift(5)
    df["close_10ago"] = df["Close"].shift(10)

    df["change_3d_pct"] = safe_div(df["Close"] - df["close_3ago"], df["close_3ago"]) * 100
    df["change_5d_pct"] = safe_div(df["Close"] - df["close_5ago"], df["close_5ago"]) * 100
    df["change_10d_pct"] = safe_div(df["Close"] - df["close_10ago"], df["close_10ago"]) * 100

    df["ma5_gap_pct"] = safe_div(df["Close"] - df["ma5"], df["ma5"]) * 100
    df["ma25_gap_pct"] = safe_div(df["Close"] - df["ma25"], df["ma25"]) * 100

    # Candle structure
    df["day_range_pct"] = safe_div(df["High"] - df["Low"], df["Low"]) * 100
    hl_range = (df["High"] - df["Low"]).replace(0, pd.NA)
    df["close_position_pct"] = safe_div(df["Close"] - df["Low"], hl_range) * 100

    body = (df["Close"] - df["Open"]).abs()
    df["body_to_range"] = safe_div(body, hl_range) * 100
    df["upper_shadow_pct"] = safe_div(df["High"] - df[["Open", "Close"]].max(axis=1), hl_range) * 100

    # Breakout context
    df["recent_high_5"] = df["High"].rolling(5).max()
    df["recent_high_20"] = df["High"].rolling(20).max()
    df["recent_low_20"] = df["Low"].rolling(20).min()

    df["breakout_gap_pct_5"] = safe_div(df["recent_high_5"] - df["Close"], df["Close"]) * 100
    df["resistance_gap_pct"] = safe_div(df["recent_high_20"] - df["Close"], df["Close"]) * 100
    df["near_breakout_5"] = (df["breakout_gap_pct_5"] <= 2).astype(int)
    df["near_breakout_20"] = (df["resistance_gap_pct"] <= 3).astype(int)

    # Liquidity
    df["turnover"] = df["Close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000
    df["turnover_avg20"] = (df["Close"] * df["Volume"]).rolling(20).mean()

    # Overheat checks
    df["rsi14"] = calc_rsi(df["Close"], 14)
    df["is_overheated"] = (
        (df["ma5_gap_pct"] >= 14)
        | (df["prev_change_pct"] >= 12)
        | (df["rsi14"] >= 78)
        | (df["day_range_pct"] >= 11)
    ).astype(int)

    # Existing-style event / absorption logic
    df["event_pre_earnings_like"] = (
        (df["vol_avg5"] > df["vol_avg20"])
        & (df["change_5d_pct"].between(5, 20))
    ).astype(int)

    df["smart_money_absorb"] = (
        (df["volume_ratio"] >= 3)
        & (df["prev_change_pct"].abs() <= 3)
        & (df["day_range_pct"] <= 6)
        & (df["close_position_pct"] >= 45)
    ).astype(int)

    # New: institutional accumulation detection
    # Idea: large turnover + above-average volume + not overheated + closes stay firm
    # despite limited day range and rising short trend.
    df["inst_accumulation"] = (
        (df["prev_day_vol_gt_20d"] == 1)
        & (df["volume_ratio"] >= 1.4)
        & (df["turnover_million"] >= 300)
        & (df["Close"] >= df["ma25"])
        & (df["ma10_slope"] > 0)
        & (df["change_5d_pct"].between(1, 15))
        & (df["prev_change_pct"].between(-2.5, 6.0))
        & (df["day_range_pct"] <= 7.5)
        & (df["close_position_pct"] >= 55)
        & (df["upper_shadow_pct"] <= 35)
    ).astype(int)

    # Stronger version for ranking
    df["inst_accumulation_strong"] = (
        (df["inst_accumulation"] == 1)
        & (df["volume_ratio"] >= 2.0)
        & (df["prev_day_vol_ratio_20"] >= 1.3)
        & (df["change_3d_pct"] >= 1.5)
        & (df["rsi14"].between(52, 72))
    ).astype(int)

    # Core signal refined
    df["core_signal"] = (
        (df["volume_ratio"] >= 1.8)
        & (df["vol_accel_3"] >= 1.25)
        & (df["close_position_pct"] >= 50)
        & (df["Close"] >= df["ma25"])
        & (df["ma5_slope"] > 0)
    ).astype(int)

    # Composite sub-scores for transparency
    df["volume_subscore"] = (
        df["volume_ratio"].clip(upper=4.0) * 2.2
        + df["vol_accel_3"].clip(upper=3.5) * 1.2
        + (df["prev_day_vol_gt_20d"] * 2.5)
    )

    df["trend_subscore"] = (
        df["change_5d_pct"].clip(lower=-5, upper=15) * 0.45
        + df["change_10d_pct"].clip(lower=-8, upper=20) * 0.15
        + (df["ma5_slope"] > 0).astype(int) * 1.5
        + (df["ma25_slope"] > 0).astype(int) * 1.5
    )

    df["structure_subscore"] = (
        (df["close_position_pct"].clip(lower=0, upper=100) / 100) * 4.0
        + (df["near_breakout_5"] * 2.0)
        + (df["near_breakout_20"] * 2.5)
        + (df["smart_money_absorb"] * 2.0)
        + (df["inst_accumulation"] * 3.0)
        + (df["inst_accumulation_strong"] * 3.0)
    )

    df["liquidity_subscore"] = 0.0
    df.loc[df["turnover_million"] >= 1000, "liquidity_subscore"] = 3.5
    df.loc[(df["turnover_million"] >= 500) & (df["turnover_million"] < 1000), "liquidity_subscore"] = 2.5
    df.loc[(df["turnover_million"] >= 200) & (df["turnover_million"] < 500), "liquidity_subscore"] = 1.2

    df["penalty_subscore"] = 0.0
    df.loc[df["ma5_gap_pct"] > 10, "penalty_subscore"] += (df["ma5_gap_pct"] - 10) * 0.8
    df.loc[df["prev_change_pct"] > 9, "penalty_subscore"] += (df["prev_change_pct"] - 9) * 0.9
    df.loc[df["day_range_pct"] > 7, "penalty_subscore"] += (df["day_range_pct"] - 7) * 0.8
    df.loc[df["close_position_pct"] < 35, "penalty_subscore"] += (35 - df["close_position_pct"]) * 0.08
    df.loc[df["rsi14"] > 76, "penalty_subscore"] += (df["rsi14"] - 76) * 0.25

    return df


def score_row(row: pd.Series) -> float:
    score = 0.0

    score += float(row.get("volume_subscore", 0))
    score += float(row.get("trend_subscore", 0))
    score += float(row.get("structure_subscore", 0))
    score += float(row.get("liquidity_subscore", 0))

    if row.get("event_pre_earnings_like", 0) == 1:
        score += 1.5
    if row.get("core_signal", 0) == 1:
        score += 2.5
    if row.get("prev_day_vol_ratio_20", 0) >= 1.5:
        score += 1.5
    if row.get("change_3d_pct", 0) >= 3:
        score += 1.2

    score -= float(row.get("penalty_subscore", 0))

    return round(score, 2)


def passes_watch_filter(latest: pd.Series) -> bool:
    if latest["turnover"] < CFG.min_turnover:
        return False
    if latest["Volume"] < CFG.min_volume:
        return False
    if latest["close_position_pct"] < CFG.min_close_position_for_watch:
        return False
    if latest["ma5_gap_pct"] > CFG.max_ma5_gap_pct:
        return False
    if latest["prev_change_pct"] > CFG.max_prev_change_pct:
        return False
    if latest["day_range_pct"] > CFG.max_day_range_pct:
        return False

    # Must have at least one meaningful trigger
    if not (
        latest["core_signal"] == 1
        or latest["smart_money_absorb"] == 1
        or latest["inst_accumulation"] == 1
        or latest["prev_day_vol_gt_20d"] == 1
        or latest["volume_ratio"] >= 2.0
    ):
        return False

    return True


def add_entry_priority(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["score", "inst_accumulation_strong", "volume_ratio"], ascending=False).reset_index(drop=True)

    df["raw_rank"] = df.index + 1
    df["entry_priority_score"] = df["score"]

    # Top name slight penalty to avoid blindly chasing 1st place,
    # while 2nd-5th often gives better fill / risk-reward at open.
    df.loc[df["raw_rank"] == 1, "entry_priority_score"] -= 1.5
    df.loc[df["raw_rank"] == 2, "entry_priority_score"] += 2.0
    df.loc[df["raw_rank"] == 3, "entry_priority_score"] += 1.2
    df.loc[df["raw_rank"].between(4, 5), "entry_priority_score"] += 0.8

    df.loc[df["inst_accumulation_strong"] == 1, "entry_priority_score"] += 2.0
    df.loc[df["prev_day_vol_gt_20d"] == 1, "entry_priority_score"] += 1.0
    df.loc[df["smart_money_absorb"] == 1, "entry_priority_score"] += 0.8
    df.loc[df["is_overheated"] == 1, "entry_priority_score"] -= 2.0

    df = df.sort_values("entry_priority_score", ascending=False).reset_index(drop=True)
    df["entry_rank"] = range(1, len(df) + 1)
    return df


def run() -> None:
    tickers = load_tickers()
    rows: list[dict] = []

    total = len(tickers)

    for i, r in tickers.iterrows():
        ticker = r["ticker"]
        name = r["name"]

        print(f"{i + 1}/{total} {ticker}")

        hist = fetch_data(ticker)
        if hist is None:
            continue

        hist = calc_indicators(hist)
        latest = hist.iloc[-1]

        if not passes_watch_filter(latest):
            time.sleep(SLEEP_SEC)
            continue

        row = {
            "ticker": ticker,
            "name": name,
            "close": round(float(latest["Close"]), 2),
            "volume": int(latest["Volume"]),
            "turnover_million": round(float(latest["turnover_million"]), 2),
            "volume_ratio": round(float(latest["volume_ratio"]), 2),
            "volume_ratio_60": round(float(latest["volume_ratio_60"]), 2),
            "vol_accel_3": round(float(latest["vol_accel_3"]), 2),
            "prev_day_vol_gt_20d": int(latest["prev_day_vol_gt_20d"]),
            "prev_day_vol_ratio_20": round(float(latest["prev_day_vol_ratio_20"]), 2),
            "close_position_pct": round(float(latest["close_position_pct"]), 2),
            "near_breakout_5": int(latest["near_breakout_5"]),
            "near_breakout_20": int(latest["near_breakout_20"]),
            "event_pre_earnings_like": int(latest["event_pre_earnings_like"]),
            "core_signal": int(latest["core_signal"]),
            "smart_money_absorb": int(latest["smart_money_absorb"]),
            "inst_accumulation": int(latest["inst_accumulation"]),
            "inst_accumulation_strong": int(latest["inst_accumulation_strong"]),
            "is_overheated": int(latest["is_overheated"]),
            "prev_change_pct": round(float(latest["prev_change_pct"]), 2),
            "change_3d_pct": round(float(latest["change_3d_pct"]), 2),
            "change_5d_pct": round(float(latest["change_5d_pct"]), 2),
            "change_10d_pct": round(float(latest["change_10d_pct"]), 2),
            "ma5_gap_pct": round(float(latest["ma5_gap_pct"]), 2),
            "ma25_gap_pct": round(float(latest["ma25_gap_pct"]), 2),
            "day_range_pct": round(float(latest["day_range_pct"]), 2),
            "resistance_gap_pct": round(float(latest["resistance_gap_pct"]), 2),
            "rsi14": round(float(latest["rsi14"]), 2),
            "volume_subscore": round(float(latest["volume_subscore"]), 2),
            "trend_subscore": round(float(latest["trend_subscore"]), 2),
            "structure_subscore": round(float(latest["structure_subscore"]), 2),
            "liquidity_subscore": round(float(latest["liquidity_subscore"]), 2),
            "penalty_subscore": round(float(latest["penalty_subscore"]), 2),
        }
        row["score"] = score_row(pd.Series(row) if "score" not in row else row)
        rows.append(row)

        time.sleep(SLEEP_SEC)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No candidates")
        return

    df = add_entry_priority(df)
    df = df.head(TOP_N_OUTPUT)

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
        "volume_ratio_60",
        "vol_accel_3",
        "prev_day_vol_gt_20d",
        "prev_day_vol_ratio_20",
        "inst_accumulation",
        "inst_accumulation_strong",
        "smart_money_absorb",
        "core_signal",
        "is_overheated",
        "close_position_pct",
        "near_breakout_5",
        "near_breakout_20",
        "prev_change_pct",
        "change_3d_pct",
        "change_5d_pct",
        "change_10d_pct",
        "ma5_gap_pct",
        "ma25_gap_pct",
        "day_range_pct",
        "resistance_gap_pct",
        "rsi14",
        "volume_subscore",
        "trend_subscore",
        "structure_subscore",
        "liquidity_subscore",
        "penalty_subscore",
    ]

    print("\n==== Morning Watchlist v3 ====")
    print(df[watch_cols].to_string(index=False))

    df[watch_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
