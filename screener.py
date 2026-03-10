import time
import pandas as pd
import yfinance as yf


TICKERS_CSV = "tickers.csv"
MAX_TICKERS = 500
BATCH_SIZE = 100
SLEEP_SEC = 0.8

# 実戦用フィルター
MIN_TURNOVER = 50_000_000   # 5000万円
MIN_VOLUME = 200_000        # 20万株

# 監視上位に残したい最低限の引け位置
MIN_CLOSE_POSITION_FOR_WATCH = 35.0


def load_tickers(csv_path=TICKERS_CSV, max_tickers=MAX_TICKERS):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["ticker"])
    df["ticker"] = df["ticker"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    return df.head(max_tickers).copy()


def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty or len(df) < 40:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        need_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in need_cols:
            if c not in df.columns:
                return None

        df = df[need_cols].copy().dropna()

        if len(df) < 40:
            return None

        return df

    except Exception:
        return None


def calc_indicators(df):
    df = df.copy()

    # 移動平均
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma25"] = df["Close"].rolling(25).mean()

    # 移動平均の向き
    df["ma5_slope"] = df["ma5"] - df["ma5"].shift(1)
    df["ma25_slope"] = df["ma25"] - df["ma25"].shift(1)

    # 出来高平均
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["vol_avg10"] = df["Volume"].rolling(10).mean()
    df["vol_avg3"] = df["Volume"].rolling(3).mean()

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

    # 5日騰落率
    df["close_5ago"] = df["Close"].shift(5)
    df["change_5d_pct"] = (df["Close"] - df["close_5ago"]) / df["close_5ago"] * 100

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

    # 直近5日高値
    df["recent_high_5"] = df["High"].rolling(5).max()

    # 5日高値までの距離
    df["breakout_gap_pct_5"] = (
        (df["recent_high_5"] - df["Close"]) / df["Close"] * 100
    )

    # 5日高値にかなり近いか（2%以内）
    df["near_breakout_5"] = (df["breakout_gap_pct_5"] <= 2.0).astype(int)

    # 売買代金
    df["turnover"] = df["Close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000

    # 決算前・イベント前っぽい動き
    df["event_volume_flag"] = (df["vol_avg3"] > df["vol_avg20"]).astype(int)
    df["event_price_flag"] = (
        (df["change_5d_pct"] >= 5) & (df["change_5d_pct"] <= 20)
    ).astype(int)
    df["event_pre_earnings_like"] = (
        (df["event_volume_flag"] == 1) & (df["event_price_flag"] == 1)
    ).astype(int)

    # 吸収型上昇（大口が静かに拾う形）
    df["smart_money_absorb"] = (
        (df["volume_ratio"] >= 3.0) &
        (df["prev_change_pct"].abs() <= 3.0) &
        (df["day_range_pct"] <= 6.0)
    ).astype(int)

    # 緩め版
    df["smart_money_absorb_loose"] = (
        (df["vol_accel_3"] >= 1.8) &
        (df["change_5d_pct"] >= 0) &
        (df["change_5d_pct"] <= 8) &
        (df["day_range_pct"] <= 8)
    ).astype(int)

    # センターピンに近い強シグナル
    df["core_signal"] = (
        (df["volume_ratio"] >= 1.8) &
        (df["vol_accel_3"] >= 1.3) &
        (df["close_position_pct"] >= 50) &
        (
            (df["near_breakout_5"] == 1) |
            (df["resistance_gap_pct"] <= 5)
        )
    ).astype(int)

    return df


def classify_row(row):
    close = row["close"]

    is_low = close <= 300
    is_large = close >= 300 and row["volume_ratio"] >= 1.2
    is_mid = row["volume_ratio"] >= 1.3

    return is_large, is_mid, is_low


def score_row(row, category):
    score = 0.0

    # ===== センターピン =====

    # 出来高倍率
    score += min(row.get("volume_ratio", 0), 10) * 2.0

    # 出来高加速
    score += min(row.get("vol_accel_3", 0), 5) * 1.8

    # ブレイク接近
    score += row.get("near_breakout_5", 0) * 3.0
    if row.get("resistance_gap_pct", 999) <= 5:
        score += 1.8

    # 引け位置（最重要補助）
    close_pos = row.get("close_position_pct", 0)
    if close_pos >= 80:
        score += 3.0
    elif close_pos >= 65:
        score += 2.0
    elif close_pos >= 50:
        score += 1.0
    elif close_pos < 25:
        score -= 2.0

    # ===== 補助 =====

    # 5日線乖離
    ma5_gap = row.get("ma5_gap_pct", 0)
    if 0 <= ma5_gap <= 12:
        score += ma5_gap * 0.5
    elif ma5_gap > 12:
        score += max(0, 12 * 0.5 - (ma5_gap - 12) * 0.8)

    # 前日比
    prev_chg = row.get("prev_change_pct", 0)
    if 0 <= prev_chg <= 12:
        score += prev_chg * 0.35
    elif prev_chg > 12:
        score += max(0, 12 * 0.35 - (prev_chg - 12) * 0.8)
    elif prev_chg < -6:
        score -= 1.5

    # 5日騰落率
    chg5 = row.get("change_5d_pct", 0)
    if 3 <= chg5 <= 18:
        score += 1.8
    elif chg5 > 25:
        score -= 2.0

    # 値幅
    day_range = row.get("day_range_pct", 0)
    if 3 <= day_range <= 12:
        score += 1.2
    elif day_range > 20:
        score -= 1.2

    # 移動平均の向き
    if row.get("ma5_slope", 0) > 0:
        score += 1.0
    if row.get("ma25_slope", 0) > 0:
        score += 0.8

    # 売買代金
    turnover_m = row.get("turnover_million", 0)
    if turnover_m >= 1000:        # 10億円以上
        score += 2.5
    elif turnover_m >= 300:       # 3億円以上
        score += 1.5
    elif turnover_m >= 100:       # 1億円以上
        score += 0.8

    # 決算前・イベント前っぽい
    score += row.get("event_pre_earnings_like", 0) * 1.5

    # 吸収型
    score += row.get("smart_money_absorb", 0) * 2.2
    score += row.get("smart_money_absorb_loose", 0) * 1.0

    # コアシグナル
    score += row.get("core_signal", 0) * 3.0

    # カテゴリ補正
    if category == "low":
        score += 0.8
    elif category == "large":
        score -= 0.3

    return round(score, 2)


def build_row_data(ticker, name, latest):
    return {
        "ticker": ticker,
        "name": name,
        "close": round(float(latest["Close"]), 3),
        "ma5": round(float(latest["ma5"]), 3),
        "ma25": round(float(latest["ma25"]), 3),
        "ma5_slope": round(float(latest["ma5_slope"]), 6),
        "ma25_slope": round(float(latest["ma25_slope"]), 6),
        "volume": int(latest["Volume"]),
        "vol_avg20": round(float(latest["vol_avg20"]), 3),
        "vol_avg10": round(float(latest["vol_avg10"]), 3),
        "vol_avg3": round(float(latest["vol_avg3"]), 3),
        "volume_ratio": round(float(latest["volume_ratio"]), 6),
        "vol_accel_3": round(float(latest["vol_accel_3"]), 6),
        "ma5_gap_pct": round(float(latest["ma5_gap_pct"]), 6),
        "prev_change_pct": round(float(latest["prev_change_pct"]), 6),
        "change_5d_pct": round(float(latest["change_5d_pct"]), 6),
        "day_range_pct": round(float(latest["day_range_pct"]), 6),
        "close_position_pct": round(float(latest["close_position_pct"]), 6),
        "recent_high_20": round(float(latest["recent_high_20"]), 3),
        "resistance_gap_pct": round(float(latest["resistance_gap_pct"]), 6),
        "recent_high_5": round(float(latest["recent_high_5"]), 3),
        "breakout_gap_pct_5": round(float(latest["breakout_gap_pct_5"]), 6),
        "near_breakout_5": int(latest["near_breakout_5"]),
        "turnover": round(float(latest["turnover"]), 0),
        "turnover_million": round(float(latest["turnover_million"]), 3),
        "event_pre_earnings_like": int(latest["event_pre_earnings_like"]),
        "smart_money_absorb": int(latest["smart_money_absorb"]),
        "smart_money_absorb_loose": int(latest["smart_money_absorb_loose"]),
        "core_signal": int(latest["core_signal"]),
    }


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

            required_cols = [
                "ma5", "ma25", "ma5_slope", "ma25_slope",
                "vol_avg20", "vol_avg10", "vol_avg3",
                "volume_ratio", "vol_accel_3",
                "ma5_gap_pct", "prev_change_pct", "change_5d_pct",
                "day_range_pct", "close_position_pct",
                "recent_high_20", "resistance_gap_pct",
                "recent_high_5", "breakout_gap_pct_5", "near_breakout_5",
                "turnover", "turnover_million",
                "event_pre_earnings_like",
                "smart_money_absorb",
                "smart_money_absorb_loose",
                "core_signal",
            ]
            if latest[required_cols].isna().any():
                continue

            row_data = build_row_data(ticker, name, latest)

            # 実戦フィルター
            if row_data["turnover"] < MIN_TURNOVER:
                continue

            if row_data["volume"] < MIN_VOLUME:
                continue

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

    large_df = pd.DataFrame(large_rows).sort_values("score", ascending=False) if large_rows else pd.DataFrame()
    mid_df = pd.DataFrame(mid_rows).sort_values("score", ascending=False) if mid_rows else pd.DataFrame()
    low_df = pd.DataFrame(low_rows).sort_values("score", ascending=False) if low_rows else pd.DataFrame()

    display_cols = [
        "ticker",
        "name",
        "category",
        "score",
        "close",
        "volume",
        "turnover_million",
        "volume_ratio",
        "vol_accel_3",
        "prev_change_pct",
        "change_5d_pct",
        "ma5_gap_pct",
        "day_range_pct",
        "close_position_pct",
        "recent_high_20",
        "resistance_gap_pct",
        "recent_high_5",
        "breakout_gap_pct_5",
        "near_breakout_5",
        "event_pre_earnings_like",
        "smart_money_absorb",
        "smart_money_absorb_loose",
        "core_signal",
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

    # 重複除外
    all_watch = pd.concat([large_df, mid_df, low_df], ignore_index=True) if (
        not large_df.empty or not mid_df.empty or not low_df.empty
    ) else pd.DataFrame()

    if not all_watch.empty:
        all_watch = all_watch.sort_values("score", ascending=False)
        morning_watch = all_watch.drop_duplicates(subset=["ticker"], keep="first").copy()

        # 引けが弱すぎるものを監視上位から落としやすくする
        morning_watch = morning_watch[
            morning_watch["close_position_pct"] >= MIN_CLOSE_POSITION_FOR_WATCH
        ].copy()

        morning_watch = morning_watch.head(10)

        print("\n==== 朝の監視ランキング（重複除外）====")
        if not morning_watch.empty:
            print(morning_watch[display_cols].to_string(index=False))
        else:
            print("該当なし")
    else:
        morning_watch = pd.DataFrame(columns=display_cols)
        print("\n==== 朝の監視ランキング（重複除外）====")
        print("該当なし")

    # CSV
    if not large_df.empty:
        large_df.to_csv("large_caps.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=display_cols).to_csv("large_caps.csv", index=False, encoding="utf-8-sig")

    if not mid_df.empty:
        mid_df.to_csv("mid_caps.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=display_cols).to_csv("mid_caps.csv", index=False, encoding="utf-8-sig")

    if not low_df.empty:
        low_df.to_csv("low_price.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=display_cols).to_csv("low_price.csv", index=False, encoding="utf-8-sig")

    morning_watch.to_csv("morning_watchlist.csv", index=False, encoding="utf-8-sig")

    print("\nCSV出力完了:")
    print(" - large_caps.csv")
    print(" - mid_caps.csv")
    print(" - low_price.csv")
    print(" - morning_watchlist.csv")


if __name__ == "__main__":
    run()
