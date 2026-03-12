import time
import pandas as pd
import yfinance as yf

TICKERS_CSV = "tickers.csv"
OUTPUT_CSV = "morning_watchlist.csv"

MAX_TICKERS = 500
BATCH_SIZE = 100
SLEEP_SEC = 0.8
TOP_N_OUTPUT = 80

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

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()].copy()

        need_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in need_cols:
            if c not in df.columns:
                return None

        df = df[need_cols].copy().dropna()

        if len(df) < 40:
            return None

        for c in need_cols:
            if isinstance(df[c], pd.DataFrame):
                df[c] = df[c].iloc[:, 0]

        return df

    except Exception as e:
        print(f"fetch_data error: {ticker} {e}")
        return None


def calc_indicators(df):
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if isinstance(df[c], pd.DataFrame):
            df[c] = df[c].iloc[:, 0]

    # 移動平均
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma10"] = df["Close"].rolling(10).mean()
    df["ma25"] = df["Close"].rolling(25).mean()

    # 移動平均の傾き
    df["ma5_slope"] = df["ma5"] - df["ma5"].shift(1)
    df["ma10_slope"] = df["ma10"] - df["ma10"].shift(1)
    df["ma25_slope"] = df["ma25"] - df["ma25"].shift(1)

    # 出来高
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["vol_avg5"] = df["Volume"].rolling(5).mean()
    df["vol_avg3"] = df["Volume"].rolling(3).mean()

    df["volume_ratio"] = df["Volume"] / df["vol_avg20"]

    recent3 = df["Volume"].rolling(3).mean()
    prev3 = recent3.shift(3)
    df["vol_accel_3"] = recent3 / prev3

    # 価格変化
    df["ma5_gap_pct"] = (df["Close"] - df["ma5"]) / df["ma5"] * 100

    df["prev_close"] = df["Close"].shift(1)
    df["prev_change_pct"] = (df["Close"] - df["prev_close"]) / df["prev_close"] * 100

    df["close_5ago"] = df["Close"].shift(5)
    df["change_5d_pct"] = (df["Close"] - df["close_5ago"]) / df["close_5ago"] * 100

    # 当日値幅
    df["day_range_pct"] = (df["High"] - df["Low"]) / df["Low"] * 100

    # 引け位置
    hl_range = (df["High"] - df["Low"]).replace(0, pd.NA)
    df["close_position_pct"] = ((df["Close"] - df["Low"]) / hl_range * 100).fillna(0)

    # ヒゲ・実体
    df["body_pct"] = (df["Close"] - df["Open"]).abs() / df["Close"] * 100

    df["lower_shadow_pct"] = (
        (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"] * 100
    ).fillna(0)

    df["upper_shadow_pct"] = (
        (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"] * 100
    ).fillna(0)

    # レジスタンス
    df["recent_high_20"] = df["High"].rolling(20).max()
    df["resistance_gap_pct"] = (df["recent_high_20"] - df["Close"]) / df["Close"] * 100

    df["recent_high_5"] = df["High"].rolling(5).max()
    df["breakout_gap_pct_5"] = (df["recent_high_5"] - df["Close"]) / df["Close"] * 100
    df["near_breakout_5"] = (df["breakout_gap_pct_5"] <= 2.0).astype(int)

    # 売買代金
    df["turnover"] = df["Close"] * df["Volume"]
    df["turnover_million"] = df["turnover"] / 1_000_000

    # 前日出来高
    df["prev_day_volume"] = df["Volume"].shift(1)
    df["prev_day_vol_gt_20d"] = (df["prev_day_volume"] > df["vol_avg20"].shift(1)).astype(int)
    df["prev_day_vol_ratio_20"] = df["prev_day_volume"] / df["vol_avg20"].shift(1)

    # イベント前っぽい形
    df["event_pre_earnings_like"] = (
        (df["vol_avg3"] > df["vol_avg20"]) &
        (df["change_5d_pct"].between(5, 20))
    ).astype(int)

    # スマートマネー吸収
    df["smart_money_absorb"] = (
        (df["volume_ratio"] >= 3.0) &
        (df["prev_change_pct"].abs() <= 3.0) &
        (df["day_range_pct"] <= 6.0)
    ).astype(int)

    # 機関仕込み（緩和版）
    df["inst_accumulation"] = (
        (df["prev_day_vol_gt_20d"] == 1) &
        (df["volume_ratio"] >= 1.2) &
        (df["turnover_million"] >= 200) &
        (df["Close"] >= df["ma25"]) &
        (df["ma10_slope"] > 0) &
        (df["change_5d_pct"].between(0.5, 16.0)) &
        (df["prev_change_pct"].between(-3.0, 7.0)) &
        (df["day_range_pct"] <= 8.5) &
        (df["close_position_pct"] >= 45) &
        (df["upper_shadow_pct"] <= 45)
    ).astype(int)

    # 強い機関仕込み
    df["inst_accumulation_strong"] = (
        (df["prev_day_vol_gt_20d"] == 1) &
        (df["prev_day_vol_ratio_20"] >= 1.5) &
        (df["volume_ratio"] >= 1.5) &
        (df["turnover_million"] >= 300) &
        (df["Close"] >= df["ma10"]) &
        (df["ma10_slope"] > 0) &
        (df["ma25_slope"] >= 0) &
        (df["change_5d_pct"].between(1.0, 12.0)) &
        (df["prev_change_pct"].between(-2.5, 5.5)) &
        (df["day_range_pct"] <= 7.5) &
        (df["close_position_pct"] >= 50) &
        (df["upper_shadow_pct"] <= 35)
    ).astype(int)

    # 吸収ローソク
    df["absorption_candle"] = (
        (df["prev_day_vol_gt_20d"] == 1) &
        (df["volume_ratio"] >= 1.2) &
        (df["turnover_million"] >= 200) &
        (df["Close"] >= df["ma25"]) &
        (df["ma10_slope"] > 0) &
        (df["prev_change_pct"].between(-1.5, 6.0)) &
        (df["day_range_pct"].between(2.0, 8.5)) &
        (df["body_pct"].between(0.8, 4.5)) &
        (df["lower_shadow_pct"] >= 1.0) &
        (df["upper_shadow_pct"] <= 3.5) &
        (df["close_position_pct"] >= 55)
    ).astype(int)

    # 強い吸収ローソク
    df["absorption_candle_strong"] = (
        (df["prev_day_vol_gt_20d"] == 1) &
        (df["prev_day_vol_ratio_20"] >= 1.5) &
        (df["volume_ratio"] >= 1.5) &
        (df["turnover_million"] >= 300) &
        (df["Close"] >= df["ma10"]) &
        (df["ma10_slope"] > 0) &
        (df["ma25_slope"] >= 0) &
        (df["prev_change_pct"].between(-1.0, 5.0)) &
        (df["day_range_pct"].between(2.5, 7.5)) &
        (df["body_pct"].between(1.0, 4.0)) &
        (df["lower_shadow_pct"] >= 1.2) &
        (df["upper_shadow_pct"] <= 2.5) &
        (df["close_position_pct"] >= 65)
    ).astype(int)

    # コアシグナル
    df["core_signal"] = (
        (df["volume_ratio"] >= 1.8) &
        (df["vol_accel_3"] >= 1.3) &
        (df["close_position_pct"] >= 50)
    ).astype(int)

    # 過熱
    df["is_overheated"] = (
        (df["ma5_gap_pct"] > 12) |
        (df["prev_change_pct"] > 12) |
        (df["day_range_pct"] > 15) |
        (df["change_5d_pct"] > 25)
    ).astype(int)

    return df


def passes_watch_filter(row):
    if row["turnover"] < MIN_TURNOVER:
        return False

    if row["Volume"] < MIN_VOLUME:
        return False

    if row["close_position_pct"] < MIN_CLOSE_POSITION_FOR_WATCH:
        return False

    # 前日出来高ブレイクは必須
    if row["prev_day_vol_gt_20d"] != 1:
        return False

    # 超過熱のみ除外
    if row["ma5_gap_pct"] > 18:
        return False

    if row["prev_change_pct"] > 18:
        return False

    if row["day_range_pct"] > 20:
        return False

    return True


def score_row(row):
    volume_subscore = 0.0
    trend_subscore = 0.0
    structure_subscore = 0.0
    liquidity_subscore = 0.0
    penalty_subscore = 0.0

    # ===== 出来高 =====
    volume_subscore += min(row["volume_ratio"], 10) * 1.8
    volume_subscore += min(row["vol_accel_3"], 5) * 1.4

    if row["prev_day_vol_gt_20d"] == 1:
        volume_subscore += 2.0

    if pd.notna(row["prev_day_vol_ratio_20"]):
        if row["prev_day_vol_ratio_20"] >= 2.0:
            volume_subscore += 2.0
        elif row["prev_day_vol_ratio_20"] >= 1.3:
            volume_subscore += 1.0

    # ===== トレンド =====
    if row["ma5_slope"] > 0:
        trend_subscore += 1.2
    if row["ma10_slope"] > 0:
        trend_subscore += 1.5
    if row["ma25_slope"] > 0:
        trend_subscore += 0.8

    if row["Close"] >= row["ma25"]:
        trend_subscore += 0.8

    if 0 <= row["change_5d_pct"] <= 15:
        trend_subscore += 2.2
    elif row["change_5d_pct"] > 20:
        penalty_subscore -= 1.5

    if 0 <= row["prev_change_pct"] <= 7:
        trend_subscore += 1.4
    elif row["prev_change_pct"] > 12:
        penalty_subscore -= 1.5

    # ===== 形 =====
    if row["near_breakout_5"] == 1:
        structure_subscore += 2.5

    if row["resistance_gap_pct"] <= 5:
        structure_subscore += 1.5

    if row["close_position_pct"] >= 80:
        structure_subscore += 2.5
    elif row["close_position_pct"] >= 65:
        structure_subscore += 1.8
    elif row["close_position_pct"] >= 50:
        structure_subscore += 0.8
    elif row["close_position_pct"] < 25:
        penalty_subscore -= 1.5

    if row["upper_shadow_pct"] <= 2.0:
        structure_subscore += 1.0
    elif row["upper_shadow_pct"] > 5.0:
        penalty_subscore -= 1.0

    if 0 <= row["ma5_gap_pct"] <= 10:
        structure_subscore += 1.8
    elif row["ma5_gap_pct"] > 12:
        penalty_subscore -= (row["ma5_gap_pct"] - 12) * 0.5

    # ===== 大口・吸収 =====
    if row["inst_accumulation"] == 1:
        structure_subscore += 4.0

    if row["inst_accumulation_strong"] == 1:
        structure_subscore += 5.0

    if row["smart_money_absorb"] == 1:
        structure_subscore += 1.8

    if row["absorption_candle"] == 1:
        structure_subscore += 2.5

    if row["absorption_candle_strong"] == 1:
        structure_subscore += 4.0

    if row["event_pre_earnings_like"] == 1:
        structure_subscore += 1.0

    if row["core_signal"] == 1:
        structure_subscore += 0.8

    # ===== 流動性 =====
    if row["turnover_million"] >= 3000:
        liquidity_subscore += 3.0
    elif row["turnover_million"] >= 1000:
        liquidity_subscore += 2.3
    elif row["turnover_million"] >= 500:
        liquidity_subscore += 1.6
    elif row["turnover_million"] >= 200:
        liquidity_subscore += 0.8

    # ===== 過熱 =====
    if row["is_overheated"] == 1:
        penalty_subscore -= 2.0

    total = (
        volume_subscore +
        trend_subscore +
        structure_subscore +
        liquidity_subscore +
        penalty_subscore
    )

    return (
        round(total, 2),
        round(volume_subscore, 2),
        round(trend_subscore, 2),
        round(structure_subscore, 2),
        round(liquidity_subscore, 2),
        round(penalty_subscore, 2),
    )


def add_entry_priority(df):
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["raw_rank"] = df.index + 1
    df["entry_priority_score"] = df["score"]

    # 上位過熱補正
    df.loc[df["raw_rank"] == 1, "entry_priority_score"] -= 1.5
    df.loc[df["raw_rank"] == 2, "entry_priority_score"] += 1.8
    df.loc[df["raw_rank"] == 3, "entry_priority_score"] += 1.2
    df.loc[df["raw_rank"].between(4, 5), "entry_priority_score"] += 0.8

    # 仕込み・吸収を優遇
    df.loc[df["inst_accumulation"] == 1, "entry_priority_score"] += 1.5
    df.loc[df["inst_accumulation_strong"] == 1, "entry_priority_score"] += 2.0
    df.loc[df["absorption_candle"] == 1, "entry_priority_score"] += 1.2
    df.loc[df["absorption_candle_strong"] == 1, "entry_priority_score"] += 1.8

    # 過熱は減点
    df.loc[df["is_overheated"] == 1, "entry_priority_score"] -= 1.5

    df = df.sort_values("entry_priority_score", ascending=False).reset_index(drop=True)
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

        if not passes_watch_filter(latest):
            continue

        score, volume_sub, trend_sub, structure_sub, liquidity_sub, penalty_sub = score_row(latest)

        row = {
            "ticker": ticker,
            "name": name,
            "close": round(float(latest["Close"]), 3),
            "volume": int(latest["Volume"]),
            "turnover_million": round(float(latest["turnover_million"]), 3),
            "volume_ratio": round(float(latest["volume_ratio"]), 6),
            "vol_accel_3": round(float(latest["vol_accel_3"]), 6),
            "prev_day_volume": int(latest["prev_day_volume"]) if pd.notna(latest["prev_day_volume"]) else None,
            "prev_day_vol_gt_20d": int(latest["prev_day_vol_gt_20d"]),
            "prev_day_vol_ratio_20": round(float(latest["prev_day_vol_ratio_20"]), 6) if pd.notna(latest["prev_day_vol_ratio_20"]) else None,
            "close_position_pct": round(float(latest["close_position_pct"]), 6),
            "upper_shadow_pct": round(float(latest["upper_shadow_pct"]), 6),
            "lower_shadow_pct": round(float(latest["lower_shadow_pct"]), 6),
            "body_pct": round(float(latest["body_pct"]), 6),
            "near_breakout_5": int(latest["near_breakout_5"]),
            "event_pre_earnings_like": int(latest["event_pre_earnings_like"]),
            "core_signal": int(latest["core_signal"]),
            "smart_money_absorb": int(latest["smart_money_absorb"]),
            "inst_accumulation": int(latest["inst_accumulation"]),
            "inst_accumulation_strong": int(latest["inst_accumulation_strong"]),
            "absorption_candle": int(latest["absorption_candle"]),
            "absorption_candle_strong": int(latest["absorption_candle_strong"]),
            "is_overheated": int(latest["is_overheated"]),
            "prev_change_pct": round(float(latest["prev_change_pct"]), 6),
            "change_5d_pct": round(float(latest["change_5d_pct"]), 6),
            "ma5_gap_pct": round(float(latest["ma5_gap_pct"]), 6),
            "day_range_pct": round(float(latest["day_range_pct"]), 6),
            "resistance_gap_pct": round(float(latest["resistance_gap_pct"]), 6),
            "ma5_slope": round(float(latest["ma5_slope"]), 6),
            "ma10_slope": round(float(latest["ma10_slope"]), 6),
            "ma25_slope": round(float(latest["ma25_slope"]), 6),
            "score": score,
            "volume_subscore": volume_sub,
            "trend_subscore": trend_sub,
            "structure_subscore": structure_sub,
            "liquidity_subscore": liquidity_sub,
            "penalty_subscore": penalty_sub,
        }

        rows.append(row)
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
        "volume_subscore",
        "trend_subscore",
        "structure_subscore",
        "liquidity_subscore",
        "penalty_subscore",
        "close",
        "turnover_million",
        "volume",
        "prev_day_volume",
        "prev_day_vol_gt_20d",
        "prev_day_vol_ratio_20",
        "volume_ratio",
        "vol_accel_3",
        "close_position_pct",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "body_pct",
        "near_breakout_5",
        "event_pre_earnings_like",
        "core_signal",
        "smart_money_absorb",
        "inst_accumulation",
        "inst_accumulation_strong",
        "absorption_candle",
        "absorption_candle_strong",
        "is_overheated",
        "prev_change_pct",
        "change_5d_pct",
        "ma5_gap_pct",
        "day_range_pct",
        "resistance_gap_pct",
        "ma5_slope",
        "ma10_slope",
        "ma25_slope",
    ]

    output_df = df[watch_cols].head(TOP_N_OUTPUT).copy()

    print("\n==== Morning Watchlist v3 ====")
    print(output_df.to_string(index=False))

    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nCSV出力完了: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
