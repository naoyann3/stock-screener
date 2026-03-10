import time
import pandas as pd
import yfinance as yf


TICKERS_CSV = "tickers.csv"
MAX_TICKERS = 500
BATCH_SIZE = 100
SLEEP_SEC = 0.8


def load_tickers(csv_path=TICKERS_CSV, max_tickers=MAX_TICKERS):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    return df.head(max_tickers).copy()


def fetch_data(ticker, period="3mo", interval="1d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty or len(df) < 30:
            return None

        # MultiIndex対策
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        need_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in need_cols:
            if c not in df.columns:
                return None

        df = df[need_cols].copy()
        df = df.dropna()
        if len(df) < 30:
            return None

        return df
    except Exception:
        return None


def calc_indicators(df):
    df = df.copy()

    # 移動平均
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma25"] = df["Close"].rolling(25).mean()

    # 出来高平均
    df["vol_avg20"] = df["Volume"].rolling(20).mean()

    # 出来高倍率
    df["volume_ratio"] = df["Volume"] / df["vol_avg20"]

    # 出来高加速（直近3日平均 ÷ その前3日平均）
    recent3 = df["Volume"].rolling(3).mean()
    prev3 = recent3.shift(3)
    df["vol_accel_3"] = recent3 / prev3

    # 5日線乖離率
    df["ma5_gap_pct"] = (df["Close"] - df["ma5"]) / df["ma5"] * 100

    # 前日終値比
    df["prev_close"] = df["Close"].shift(1)
    df["prev_change_pct"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100

    # 当日値幅率
    df["day_range_pct"] = (df["High"] - df["Low"]) / df["Low"] * 100

    # 引け位置（0=安値引け, 100=高値引け）
    hl_range = (df["High"] - df["Low"]).replace(0, pd.NA)
    df["close_position_pct"] = (df["Close"] - df["Low"]) / hl_range * 100
    df["close_position_pct"] = df["close_position_pct"].fillna(0)

    # 直近20日高値
    df["recent_high_20"] = df["High"].rolling(20).max()

    # 上値余地
    df["resistance_gap_pct"] = (
        (df["recent_high_20"] - df["Close"]) / df["Close"] * 100
    )

    # ===== 追加部分：直近5日高値ブレイク判定 =====
    # 直近5日高値（当日含む）
    df["recent_high_5"] = df["High"].rolling(5).max()

    # 5日高値までの距離
    df["breakout_gap_pct_5"] = (
        (df["recent_high_5"] - df["Close"]) / df["Close"] * 100
    )

    # 5日高値にかなり近いか（2%以内なら1）
    df["near_breakout_5"] = (df["breakout_gap_pct_5"] <= 2.0).astype(int)

    return df


def classify_row(row):
    close = row["close"]

    is_low = close <= 300
    is_large = close >= 300 and row["volume_ratio"] >= 1.2
    is_mid = row["volume_ratio"] >= 1.3

    return is_large, is_mid, is_low


def score_row(row, category):
    score = 0.0

    # 出来高系
    score += min(row.get("volume_ratio", 0), 10) * 2.0
    score += min(row.get("vol_accel_3", 0), 5) * 1.5

    # トレンド系
    ma5_gap = row.get("ma5_gap_pct", 0)
    if 0 <= ma5_gap <= 15:
        score += ma5_gap * 0.6
    elif ma5_gap > 15:
        score += max(0, 15 * 0.6 - (ma5_gap - 15) * 0.8)

    # 前日比
    prev_chg = row.get("prev_change_pct", 0)
    if 0 <= prev_chg <= 15:
        score += prev_chg * 0.5
    elif prev_chg > 15:
        score += max(0, 15 * 0.5 - (prev_chg - 15) * 0.7)

    # 引け位置
    close_pos = row.get("close_position_pct", 0)
    if close_pos >= 70:
        score += 2.5
    elif close_pos >= 50:
        score += 1.0
    else:
        score -= 1.0

    # 当日値幅
    day_range = row.get("day_range_pct", 0)
    if 3 <= day_range <= 12:
        score += 1.5
    elif day_range > 20:
        score -= 1.0

    # 20日高値まで近いなら加点
    if row.get("resistance_gap_pct", 999) <= 5:
        score += 1.5

    # ===== 追加部分：5日高値ブレイク直前 =====
    score += row.get("near_breakout_5", 0) * 2.0

    # カテゴリ補正
    if category == "low":
        score += 1.0
    elif category == "large":
        score -= 0.5

    return round(score, 2)


def build_row_data(ticker, name, latest):
    row_data = {
        "ticker": ticker,
        "name": name,
        "close": round(float(latest["Close"]), 3),
        "ma5": round(float(latest["ma5"]), 3),
        "ma25": round(float(latest["ma25"]), 3),
        "volume": int(latest["Volume"]),
        "vol_avg20": round(float(latest["vol_avg20"]), 3),
        "volume_ratio": round(float(latest["volume_ratio"]), 6),
        "vol_accel_3": round(float(latest["vol_accel_3"]), 6),
        "ma5_gap_pct": round(float(latest["ma5_gap_pct"]), 6),
        "prev_change_pct": round(float(latest["prev_change_pct"]), 6),
        "day_range_pct": round(float(latest["day_range_pct"]), 6),
        "close_position_pct": round(float(latest["close_position_pct"]), 6),
        "recent_high_20": round(float(latest["recent_high_20"]), 3),
        "resistance_gap_pct": round(float(latest["resistance_gap_pct"]), 6),
        "recent_high_5": round(float(latest["recent_high_5"]), 3),
        "breakout_gap_pct_5": round(float(latest["breakout_gap_pct_5"]), 6),
        "near_breakout_5": int(latest["near_breakout_5"]),
    }
    return row_data


def run():
    tickers_df = load_tickers()
    total = len(tickers_df)

    large_rows = []
    mid_rows = []
    low_rows = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        print(f"\n===== Batch {batch_start+1}-{batch_end}/{total} =====")

        batch = tickers_df.iloc[batch_start:batch_end]

        for idx, (_, row) in enumerate(batch.iterrows(), start=batch_start + 1):
            ticker = row["ticker"]
            name = str(row["name"])
            print(f"[{idx}/{total}] Checking {ticker} {name}")

            hist = fetch_data(ticker)
            if hist is None:
                continue

            hist = calc_indicators(hist)
            latest = hist.iloc[-1]

            # 必須指標がNaNならスキップ
            required_cols = [
                "ma5", "ma25", "vol_avg20", "volume_ratio", "vol_accel_3",
                "ma5_gap_pct", "prev_change_pct", "day_range_pct",
                "close_position_pct", "recent_high_20", "resistance_gap_pct",
                "recent_high_5", "breakout_gap_pct_5", "near_breakout_5"
            ]
            if latest[required_cols].isna().any():
                continue

            row_data = build_row_data(ticker, name, latest)
            is_large, is_mid, is_low = classify_row(row_data)

            if is_large:
                item = row_data.copy()
                item["category"] = "large"
                item["score"] = score_row(item, "large")
                large_rows.append(item)

            if is_mid:
                item = row_data.copy()
                item["category"] = "mid"
                item["score"] = score_row(item, "mid")
                mid_rows.append(item)

            if is_low:
                item = row_data.copy()
                item["category"] = "low"
                item["score"] = score_row(item, "low")
                low_rows.append(item)

            time.sleep(SLEEP_SEC)

    large_df = pd.DataFrame(large_rows).sort_values("score", ascending=False)
    mid_df = pd.DataFrame(mid_rows).sort_values("score", ascending=False)
    low_df = pd.DataFrame(low_rows).sort_values("score", ascending=False)

    display_cols = [
        "ticker",
        "name",
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
        "recent_high_5",
        "breakout_gap_pct_5",
        "near_breakout_5",
        "category",
        "score",
    ]

    print("\n==== 大型テーマ株（上位10件）====")
    if not large_df.empty:
        print(large_df[display_cols].head(10).to_string(index=False))
    else:
        print("該当なし")

    print("\n==== 中小型テーマ株（上位10件）====")
    if not mid_df.empty:
        print(mid_df[display_cols].head(10).to_string(index=False))
    else:
        print("該当なし")

    print("\n==== 低位株候補（上位10件）====")
    if not low_df.empty:
        print(low_df[display_cols].head(10).to_string(index=False))
    else:
        print("該当なし")

    # 朝の監視ランキング（重複除外）
    all_watch = pd.concat([large_df, mid_df, low_df], ignore_index=True)
    if not all_watch.empty:
        all_watch = all_watch.sort_values("score", ascending=False)
        morning_watch = all_watch.drop_duplicates(subset=["ticker"], keep="first")
        morning_watch = morning_watch.head(10)

        print("\n==== 朝の監視ランキング（重複除外）====")
        print(morning_watch[display_cols].to_string(index=False))
    else:
        morning_watch = pd.DataFrame(columns=display_cols)
        print("\n==== 朝の監視ランキング（重複除外）====")
        print("該当なし")

    # CSV出力
    large_df.to_csv("large_caps.csv", index=False, encoding="utf-8-sig")
    mid_df.to_csv("mid_caps.csv", index=False, encoding="utf-8-sig")
    low_df.to_csv("low_price.csv", index=False, encoding="utf-8-sig")
    morning_watch.to_csv("morning_watchlist.csv", index=False, encoding="utf-8-sig")

    print("\nCSV出力完了:")
    print(" - large_caps.csv")
    print(" - mid_caps.csv")
    print(" - low_price.csv")
    print(" - morning_watchlist.csv")


if __name__ == "__main__":
    run()
