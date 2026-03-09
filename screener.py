import os
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf


# =========================
# 設定
# =========================
TICKERS_FILE = "tickers.csv"
LIMIT = 500           # 上から何件チェックするか
BATCH_SIZE = 100      # 100件ずつ区切って処理
HISTORY_DAYS = 60     # 取得日数
MIN_DAYS = 30         # 最低必要本数

LOW_PRICE_THRESHOLD = 300  # 低位株判定
TOP_N = 10                 # 各ランキング表示件数


# =========================
# 補助関数
# =========================
def safe_pct(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return None
    return numerator / denominator * 100


def calc_close_position(high_price, low_price, close_price):
    """
    当日のレンジの中で終値がどこにあるか（％）
    0   = 安値引け
    100 = 高値引け
    """
    if pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price):
        return None
    if high_price == low_price:
        return 50.0
    return (close_price - low_price) / (high_price - low_price) * 100


def score_row(row, category):
    """
    スコア計算
    category によって少し補正
    """
    volume_ratio = row.get("volume_ratio", 0) or 0
    vol_accel_3 = row.get("vol_accel_3", 0) or 0
    ma5_gap_pct = row.get("ma5_gap_pct", 0) or 0
    prev_change_pct = row.get("prev_change_pct", 0) or 0
    close_position_pct = row.get("close_position_pct", 0) or 0
    resistance_gap_pct = row.get("resistance_gap_pct", 0) or 0

    # 基本スコア
    score = (
        volume_ratio * 2.0
        + vol_accel_3 * 1.5
        + ma5_gap_pct * 0.45
        + prev_change_pct * 0.25
        + (close_position_pct / 100.0) * 1.2
    )

    # 上値余地がある銘柄をやや加点
    if resistance_gap_pct is not None:
        if 0 < resistance_gap_pct <= 12:
            score += 1.2
        elif 12 < resistance_gap_pct <= 25:
            score += 0.6

    # カテゴリ補正
    if category == "low":
        score += 1.0
    elif category == "large":
        score -= 0.3

    return round(score, 2)


def print_df(title, df, top_n=TOP_N):
    print(f"\n==== {title}（上位{top_n}件）====")
    if df.empty:
        print("該当なし")
    else:
        print(df.head(top_n).to_string(index=False))


# =========================
# データ取得と判定
# =========================
def load_tickers():
    df = pd.read_csv(TICKERS_FILE, encoding="utf-8-sig")

    # 想定列:
    # ticker, name
    required_cols = {"ticker", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{TICKERS_FILE} に必要な列 {required_cols} がありません。")

    df = df.dropna(subset=["ticker"]).copy()
    df["ticker"] = df["ticker"].astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    return df


def fetch_one(ticker):
    """
    1銘柄の株価データ取得
    """
    try:
        data = yf.download(
            ticker,
            period=f"{HISTORY_DAYS}d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if data is None or data.empty or len(data) < MIN_DAYS:
            return None

        # MultiIndex対策
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.copy()

        # 数値列を保証
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in data.columns:
                return None
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data = data.dropna(subset=["Close", "Volume"])

        if len(data) < MIN_DAYS:
            return None

        # テクニカル
        data["ma5"] = data["Close"].rolling(5).mean()
        data["ma25"] = data["Close"].rolling(25).mean()
        data["vol_avg20"] = data["Volume"].rolling(20).mean()
        data["recent_high_20"] = data["High"].rolling(20).max()
        data["vol_accel_3"] = data["Volume"].rolling(3).mean() / data["Volume"].rolling(20).mean()

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        close_price = float(latest["Close"])
        prev_close = float(prev["Close"])
        open_price = float(latest["Open"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])
        volume = float(latest["Volume"])
        ma5 = float(latest["ma5"]) if pd.notna(latest["ma5"]) else None
        ma25 = float(latest["ma25"]) if pd.notna(latest["ma25"]) else None
        vol_avg20 = float(latest["vol_avg20"]) if pd.notna(latest["vol_avg20"]) else None
        recent_high_20 = float(latest["recent_high_20"]) if pd.notna(latest["recent_high_20"]) else None
        vol_accel_3 = float(latest["vol_accel_3"]) if pd.notna(latest["vol_accel_3"]) else None

        if ma5 is None or ma25 is None or vol_avg20 is None or recent_high_20 is None or vol_accel_3 is None:
            return None

        volume_ratio = volume / vol_avg20 if vol_avg20 and vol_avg20 > 0 else None
        ma5_gap_pct = safe_pct(close_price - ma5, ma5)
        prev_change_pct = safe_pct(close_price - prev_close, prev_close)
        day_range_pct = safe_pct(high_price - low_price, low_price)
        close_position_pct = calc_close_position(high_price, low_price, close_price)
        resistance_gap_pct = safe_pct(recent_high_20 - close_price, close_price)

        result = {
            "ticker": ticker,
            "close": round(close_price, 2),
            "ma5": round(ma5, 2),
            "ma25": round(ma25, 2),
            "volume": int(volume),
            "vol_avg20": round(vol_avg20, 2),
            "volume_ratio": round(volume_ratio, 6) if volume_ratio is not None else None,
            "ma5_gap_pct": round(ma5_gap_pct, 6) if ma5_gap_pct is not None else None,
            "prev_change_pct": round(prev_change_pct, 6) if prev_change_pct is not None else None,
            "day_range_pct": round(day_range_pct, 6) if day_range_pct is not None else None,
            "close_position_pct": round(close_position_pct, 6) if close_position_pct is not None else None,
            "recent_high_20": round(recent_high_20, 2),
            "resistance_gap_pct": round(resistance_gap_pct, 6) if resistance_gap_pct is not None else None,
            "vol_accel_3": round(vol_accel_3, 6),
        }
        return result

    except Exception:
        return None


def is_candidate(row):
    """
    基本候補条件
    """
    if row["ma5"] is None or row["ma25"] is None:
        return False

    return (
        row["close"] > row["ma5"] and
        row["volume_ratio"] is not None and row["volume_ratio"] >= 1.2 and
        row["vol_accel_3"] is not None and row["vol_accel_3"] >= 1.1
    )


def is_large_theme(row):
    """
    大型テーマ株
    """
    if not is_candidate(row):
        return False
    return row["close"] >= 300


def is_mid_theme(row):
    """
    中小型テーマ株
    """
    if not is_candidate(row):
        return False
    return True


def is_low_price(row):
    """
    低位株候補
    """
    if not is_candidate(row):
        return False
    return row["close"] <= LOW_PRICE_THRESHOLD


# =========================
# メイン処理
# =========================
def run():
    ticker_df = load_tickers()
    ticker_df = ticker_df.head(LIMIT).copy()

    large_rows = []
    mid_rows = []
    low_rows = []

    total = len(ticker_df)

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        print(f"\n===== Batch {batch_start + 1}-{batch_end}/{total} =====")

        batch = ticker_df.iloc[batch_start:batch_end]

        for i, (_, row_info) in enumerate(batch.iterrows(), start=batch_start + 1):
            ticker = row_info["ticker"]
            name = row_info["name"]
            print(f"[{i}/{total}] Checking {ticker} {name}")

            result = fetch_one(ticker)
            if result is None:
                continue

            result["name"] = name

            if is_large_theme(result):
                result_large = result.copy()
                result_large["category"] = "large"
                result_large["score"] = score_row(result_large, "large")
                large_rows.append(result_large)

            if is_mid_theme(result):
                result_mid = result.copy()
                result_mid["category"] = "mid"
                result_mid["score"] = score_row(result_mid, "mid")
                mid_rows.append(result_mid)

            if is_low_price(result):
                result_low = result.copy()
                result_low["category"] = "low"
                result_low["score"] = score_row(result_low, "low")
                low_rows.append(result_low)

    # DataFrame化
    large_df = pd.DataFrame(large_rows)
    mid_df = pd.DataFrame(mid_rows)
    low_df = pd.DataFrame(low_rows)

    # 並び順
    sort_cols = ["score", "volume_ratio", "vol_accel_3", "prev_change_pct"]
    ascending = [False, False, False, False]

    if not large_df.empty:
        large_df = large_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if not mid_df.empty:
        mid_df = mid_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    if not low_df.empty:
        low_df = low_df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    # 朝の監視ランキング（重複除外）
    watch_df = pd.concat([large_df, mid_df, low_df], ignore_index=True)

    if not watch_df.empty:
        watch_df = watch_df.sort_values(
            ["score", "volume_ratio", "vol_accel_3", "prev_change_pct"],
            ascending=[False, False, False, False]
        ).drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    # 表示したい列順
    display_cols = [
        "ticker",
        "close",
        "ma5",
        "ma25",
        "volume",
        "vol_avg20",
        "volume_ratio",
        "vol_accel_3",
        "ma5_gap_pct",
        "prev_change_pct",
        "day_range_pct",
        "close_position_pct",
        "recent_high_20",
        "resistance_gap_pct",
        "name",
        "category",
        "score",
    ]

    for df in [large_df, mid_df, low_df, watch_df]:
        if not df.empty:
            missing = [c for c in display_cols if c not in df.columns]
            for c in missing:
                df[c] = None

    large_df = large_df[display_cols] if not large_df.empty else pd.DataFrame(columns=display_cols)
    mid_df = mid_df[display_cols] if not mid_df.empty else pd.DataFrame(columns=display_cols)
    low_df = low_df[display_cols] if not low_df.empty else pd.DataFrame(columns=display_cols)
    watch_df = watch_df[display_cols] if not watch_df.empty else pd.DataFrame(columns=display_cols)

    # 表示
    print_df("大型テーマ株", large_df)
    print_df("中小型テーマ株", mid_df)
    print_df("低位株候補", low_df)

    print("\n==== 朝の監視ランキング（重複除外）====")
    if watch_df.empty:
        print("該当なし")
    else:
        print(watch_df.head(TOP_N).to_string(index=False))

    # 通常出力
    large_df.to_csv("large_caps.csv", index=False, encoding="utf-8-sig")
    mid_df.to_csv("mid_caps.csv", index=False, encoding="utf-8-sig")
    low_df.to_csv("low_price.csv", index=False, encoding="utf-8-sig")
    watch_df.to_csv("morning_watchlist.csv", index=False, encoding="utf-8-sig")

    # 履歴保存
    run_date = os.getenv("RUN_DATE")
    if not run_date:
        run_date = datetime.now().strftime("%Y-%m-%d")

    history_dir = Path("history")
    history_dir.mkdir(exist_ok=True)

    large_df.to_csv(history_dir / f"large_caps_{run_date}.csv", index=False, encoding="utf-8-sig")
    mid_df.to_csv(history_dir / f"mid_caps_{run_date}.csv", index=False, encoding="utf-8-sig")
    low_df.to_csv(history_dir / f"low_price_{run_date}.csv", index=False, encoding="utf-8-sig")
    watch_df.to_csv(history_dir / f"morning_watchlist_{run_date}.csv", index=False, encoding="utf-8-sig")

    print("\nCSV出力完了:")
    print(" - large_caps.csv")
    print(" - mid_caps.csv")
    print(" - low_price.csv")
    print(" - morning_watchlist.csv")
    print(f" - history/large_caps_{run_date}.csv")
    print(f" - history/mid_caps_{run_date}.csv")
    print(f" - history/low_price_{run_date}.csv")
    print(f" - history/morning_watchlist_{run_date}.csv")


if __name__ == "__main__":
    run()
