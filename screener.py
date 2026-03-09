import time
import gc
import pandas as pd
import yfinance as yf


def load_tickers(path="tickers.csv"):
    df = pd.read_csv(path)
    return df[["ticker", "name"]].to_dict("records")


def analyze_stock(ticker):
    try:
        df = yf.download(ticker, period="3mo", progress=False, auto_adjust=False)

        if df.empty or len(df) < 30:
            return None

        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]
        low = df["Low"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(volume, pd.DataFrame):
            volume = volume.iloc[:, 0]
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]

        ma5 = close.rolling(5).mean()
        ma25 = close.rolling(25).mean()
        vol_avg20 = volume.rolling(20).mean()

        # 直近20日高値
        recent_high_20 = high.rolling(20).max()

        last_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        last_high = high.iloc[-1]
        last_low = low.iloc[-1]
        last_ma5 = ma5.iloc[-1]
        last_ma25 = ma25.iloc[-1]
        last_volume = volume.iloc[-1]
        last_vol_avg20 = vol_avg20.iloc[-1]
        last_recent_high_20 = recent_high_20.iloc[-1]

        if (
            pd.isna(last_ma5)
            or pd.isna(last_ma25)
            or pd.isna(last_vol_avg20)
            or pd.isna(last_recent_high_20)
        ):
            return None

        # 出来高倍率
        volume_ratio = last_volume / last_vol_avg20 if last_vol_avg20 > 0 else 0

        # 5日線乖離率
        ma5_gap_pct = ((last_close / last_ma5) - 1) * 100 if last_ma5 > 0 else 0

        # 前日終値比
        prev_change_pct = ((last_close / prev_close) - 1) * 100 if prev_close > 0 else 0

        # 1日値幅率
        day_range_pct = ((last_high - last_low) / prev_close) * 100 if prev_close > 0 else 0

        # 引け位置率（安値=0、高値=100）
        if last_high > last_low:
            close_position_pct = ((last_close - last_low) / (last_high - last_low)) * 100
        else:
            close_position_pct = 50.0

        # 直近20日高値までの上値余地
        resistance_gap_pct = (
            ((last_recent_high_20 - last_close) / last_close) * 100
            if last_close > 0
            else 0
        )

        # 20日高値ブレイク直前度（小さいほど高値に近い）
        near_breakout = (
            (last_recent_high_20 - last_close) / last_close
            if last_close > 0
            else 999
        )

        result = {
            "ticker": ticker,
            "close": float(last_close),
            "ma5": float(last_ma5),
            "ma25": float(last_ma25),
            "volume": int(last_volume),
            "vol_avg20": float(last_vol_avg20),
            "volume_ratio": float(volume_ratio),
            "ma5_gap_pct": float(ma5_gap_pct),
            "prev_change_pct": float(prev_change_pct),
            "day_range_pct": float(day_range_pct),
            "close_position_pct": float(close_position_pct),
            "recent_high_20": float(last_recent_high_20),
            "resistance_gap_pct": float(resistance_gap_pct),
            "near_breakout": float(near_breakout),
        }

        del df
        gc.collect()
        return result

    except Exception:
        return None


def calc_score(data, category):
    volume_ratio = data["volume_ratio"]
    ma5_gap_pct = data["ma5_gap_pct"]
    resistance_gap_pct = data["resistance_gap_pct"]
    near_breakout = data["near_breakout"]
    close_position_pct = data["close_position_pct"]

    overheat_penalty = 0
    if ma5_gap_pct > 15:
        overheat_penalty = (ma5_gap_pct - 15) * 0.5

    if category == "large":
        score = volume_ratio * 2.0 + ma5_gap_pct * 0.5 - overheat_penalty
    elif category == "mid":
        score = volume_ratio * 2.5 + ma5_gap_pct * 0.7 - overheat_penalty
    elif category == "low":
        score = volume_ratio * 3.0 + ma5_gap_pct * 0.5 - overheat_penalty
    else:
        score = volume_ratio * 2.0 + ma5_gap_pct * 0.5 - overheat_penalty

    # 上値余地に少し加点（最大+5点）
    score += min(resistance_gap_pct / 5, 5)

    # 20日高値が近い銘柄を加点
    # 3%以内:+3点 / 5%以内:+2点 / 8%以内:+1点
    if near_breakout <= 0.03:
        score += 3
    elif near_breakout <= 0.05:
        score += 2
    elif near_breakout <= 0.08:
        score += 1

    # 引け位置が高いほど少し加点
    if close_position_pct >= 80:
        score += 1.5
    elif close_position_pct >= 60:
        score += 0.8

    return round(score, 2)


def classify_stock(data, large_caps, mid_caps, low_price):
    close = data["close"]
    ma5 = data["ma5"]
    ma25 = data["ma25"]
    volume = data["volume"]
    vol_avg20 = data["vol_avg20"]

    # 大型テーマ株
    cond1 = close > ma5
    cond2 = ma5 > ma25
    cond3 = volume > vol_avg20 * 1.2
    if cond1 and cond2 and cond3:
        d = data.copy()
        d["category"] = "large"
        d["score"] = calc_score(d, "large")
        large_caps.append(d)

    # 中小型テーマ株
    cond4 = close > ma5
    cond5 = volume > vol_avg20 * 1.1
    if cond4 and cond5:
        d = data.copy()
        d["category"] = "mid"
        d["score"] = calc_score(d, "mid")
        mid_caps.append(d)

    # 低位株候補
    cond6 = close < 300
    cond7 = volume > vol_avg20 * 1.5
    cond8 = close > ma5
    if cond6 and cond7 and cond8:
        d = data.copy()
        d["category"] = "low"
        d["score"] = calc_score(d, "low")
        low_price.append(d)


def run():
    tickers = load_tickers()[:500]
    batch_size = 100

    large_caps = []
    mid_caps = []
    low_price = []

    total = len(tickers)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = tickers[batch_start:batch_end]

        print(f"\n===== Batch {batch_start+1}-{batch_end}/{total} =====")

        for i, item in enumerate(batch, start=batch_start + 1):
            ticker = item["ticker"]
            name = item["name"]

            print(f"[{i}/{total}] Checking {ticker} {name}")
            data = analyze_stock(ticker)

            if data is not None:
                data["name"] = name
                classify_stock(data, large_caps, mid_caps, low_price)

            time.sleep(0.1)

        gc.collect()
        time.sleep(1)

    df_large = pd.DataFrame(large_caps)
    df_mid = pd.DataFrame(mid_caps)
    df_low = pd.DataFrame(low_price)

    columns = [
        "ticker",
        "name",
        "category",
        "score",
        "close",
        "ma5",
        "ma25",
        "volume",
        "vol_avg20",
        "volume_ratio",
        "ma5_gap_pct",
        "prev_change_pct",
        "day_range_pct",
        "close_position_pct",
        "recent_high_20",
        "resistance_gap_pct",
        "near_breakout",
    ]

    if df_large.empty:
        df_large = pd.DataFrame(columns=columns)
    else:
        df_large = df_large.sort_values(by="score", ascending=False)

    if df_mid.empty:
        df_mid = pd.DataFrame(columns=columns)
    else:
        df_mid = df_mid.sort_values(by="score", ascending=False)

    if df_low.empty:
        df_low = pd.DataFrame(columns=columns)
    else:
        df_low = df_low.sort_values(by="score", ascending=False)

    print("\n==== 大型テーマ株（上位10件）====")
    print(df_large.head(10).to_string(index=False))

    print("\n==== 中小型テーマ株（上位10件）====")
    print(df_mid.head(10).to_string(index=False))

    print("\n==== 低位株候補（上位10件）====")
    print(df_low.head(10).to_string(index=False))

    # 朝の監視リストを作成
    watch_low = df_low.head(5)
    watch_mid = df_mid.head(5)
    watch_large = df_large.head(3)

    morning_watchlist = pd.concat([watch_low, watch_mid, watch_large], ignore_index=True)
    morning_watchlist = morning_watchlist.sort_values(by="score", ascending=False)

    # 同じ銘柄が複数カテゴリに入る場合、スコアが高いものだけ残す
    morning_watchlist = morning_watchlist.drop_duplicates(subset=["ticker"], keep="first")

    print("\n==== 朝の監視ランキング（重複除外）====")
    print(morning_watchlist.to_string(index=False))

    df_large.to_csv("large_caps.csv", index=False, encoding="utf-8-sig")
    df_mid.to_csv("mid_caps.csv", index=False, encoding="utf-8-sig")
    df_low.to_csv("low_price.csv", index=False, encoding="utf-8-sig")
    morning_watchlist.to_csv("morning_watchlist.csv", index=False, encoding="utf-8-sig")

    print("\nCSV出力完了:")
    print(" - large_caps.csv")
    print(" - mid_caps.csv")
    print(" - low_price.csv")
    print(" - morning_watchlist.csv")


if __name__ == "__main__":
    run()