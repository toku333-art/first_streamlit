# ===== Streamlit: 株価売買シミュレーションアプリ =====
import re
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import streamlit as st

# ---------------- UI 基本設定 ----------------
st.set_page_config(page_title="株価売買シミュレーター", layout="wide")
st.title("株価売買シミュレーションアプリ")

st.sidebar.write("""
### 売買シュミレーター
指定した銘柄の**ゴールデンクロス/デッドクロス(MA 5/25)** を用いた
シンプルな戦略でバックテストを行います。
""")

# ---------------- サイドバー入力 ----------------
symbol_in = st.sidebar.text_input("銘柄コード（例: 1928 / 7203 / AAPL など）", value="1928")
months = st.sidebar.slider("期間(月)", min_value=1, max_value=36, value=6, step=1)
init_cash = st.sidebar.number_input("資本金(円)", min_value=100_000, max_value=10_000_000, value=1_000_000, step=10_000)
commission_pct = st.sidebar.slider("売買手数料(%)", min_value=0.0, max_value=5.0, value=0.10, step=0.01, help="片道の手数料率。往復で約2倍のコストになります。")
run_bt = st.sidebar.button("シミュレーション実行", type="primary")

# ---------------- ユーティリティ ----------------
def normalize_ticker(user_text: str) -> str:
    """数値のみは TSE 想定で .T を付与。AAPL 等はそのまま。"""
    parts = [p for p in re.split(r"[\s,，、]+", str(user_text).strip()) if p]
    if not parts:
        raise ValueError("銘柄が空です")
    s = parts[0].upper()
    return f"{s}.T" if s.isdigit() else s

def choose_yf_period(months: int) -> str:
    m = int(months)
    if m <= 6:   return "1y"
    if m <= 12:  return "2y"
    if m <= 24:  return "5y"
    return "10y"

@st.cache_data(show_spinner=False, ttl=60*30)
def fetch_company_name(ticker: str) -> str:
    try:
        tk = yf.Ticker(ticker)
        info = tk.get_info()
        for key in ("shortName", "longName", "name"):
            if isinstance(info, dict) and info.get(key):
                return str(info[key])
    except Exception:
        pass
    return ticker

@st.cache_data(show_spinner=True, ttl=60*30)
def download_ohlcv(ticker: str, period="2y", interval="1d") -> pd.DataFrame:
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw is None or getattr(raw, "empty", True):
        raise ValueError(f"データが取得できませんでした: {ticker} (period={period}, interval={interval})")

    def _pick_one(df_or_series, key):
        if key not in df_or_series.columns:
            return None
        obj = df_or_series[key]
        return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj

    df = raw.copy()
    # マルチカラム対策
    if isinstance(df.columns, pd.MultiIndex):
        try:
            flds = ["Open","High","Low","Close","Adj Close","Volume"]
            sub = {f: df[(f, df.columns.levels[1][0])] for f in flds
                   if (f, df.columns.levels[1][0]) in df.columns}
            df = pd.DataFrame(sub) if sub else df[df.columns.levels[0][0]]
        except Exception:
            sw = df.swaplevel(axis=1).sort_index(axis=1)
            df = sw[sw.columns.levels[0][0]]

    open_s  = _pick_one(df, "Open")
    high_s  = _pick_one(df, "High")
    low_s   = _pick_one(df, "Low")
    close_s = _pick_one(df, "Close")
    if close_s is None:
        close_s = _pick_one(df, "Adj Close")
    vol_s   = _pick_one(df, "Volume")

    needed = {"Open": open_s, "High": high_s, "Low": low_s, "Close": close_s, "Volume": vol_s}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise ValueError(f"必要列が見つかりません: {missing}. 取得列: {list(df.columns)}")

    out = pd.DataFrame({
        "Open":   pd.to_numeric(open_s,  errors="coerce"),
        "High":   pd.to_numeric(high_s,  errors="coerce"),
        "Low":    pd.to_numeric(low_s,   errors="coerce"),
        "Close":  pd.to_numeric(close_s, errors="coerce"),
        "Volume": pd.to_numeric(vol_s,   errors="coerce"),
    }).dropna(how="any")

    if out.empty:
        raise ValueError(f"有効なOHLCVが得られませんでした: {ticker}")

    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out.sort_index()

# ---------------- 戦略定義（backtesting.py） ----------------
def SMA(arr, n: int):
    s = pd.Series(np.asarray(arr, dtype=float))
    return s.rolling(n, min_periods=n).mean().to_numpy()

def _get_cash(strategy):
    if hasattr(strategy, "broker") and hasattr(strategy.broker, "cash"):
        return float(strategy.broker.cash)
    if hasattr(strategy, "_broker") and hasattr(strategy._broker, "cash"):
        return float(strategy._broker.cash)
    if hasattr(strategy, "cash"):
        return float(strategy.cash)
    return None

class GoldenDeadCrossStrategy(Strategy):
    ma_fast = 5
    ma_slow = 25
    lot_round = 1  # 使わないが互換のため残す

    def init(self):
        close = self.data.Close
        self.sma_fast = self.I(SMA, close, self.ma_fast)
        self.sma_slow = self.I(SMA, close, self.ma_slow)

    def next(self):
        # デッドクロスでクローズ（fast が slow を下抜け）
        if self.position and cross_down(self.sma_fast, self.sma_slow):
            self.position.close()

        # ゴールデンクロスでエントリ（fast が slow を上抜け）
        if not self.position and cross_up(self.sma_fast, self.sma_slow):
            price = float(self.data.Close[-1])
            cash  = _get_cash(self)
            if cash is None or not np.isfinite(cash):
                size = 1
            else:
                max_shares = int(cash // price)
                size = max(1, max_shares)  # ← 必ず1株以上で発注
            self.buy(size=size)


        # 置き換え後（等号を含む判定を使用）
        if self.position and cross_down(self.sma_slow, self.sma_fast):
            self.position.close()

        if not self.position and cross_up(self.sma_fast, self.sma_slow):
            price = float(self.data.Close[-1])
            cash  = _get_cash(self)
            if cash is None or not np.isfinite(cash):
                size = 1
            else:
                max_shares = int(cash // price)
                size = max(1, max_shares)   # ←①の修正
            self.buy(size=size)

def cross_up(fast, slow):
    # 前足で fast <= slow かつ 今足で fast > slow
    if np.isnan(fast[-2]) or np.isnan(slow[-2]) or np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return False
    return fast[-2] <= slow[-2] and fast[-1] > slow[-1]

def cross_down(fast, slow):
    # 前足で fast >= slow かつ 今足で fast < slow
    if np.isnan(fast[-2]) or np.isnan(slow[-2]) or np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return False
    return fast[-2] >= slow[-2] and fast[-1] < slow[-1]


# ---------------- チャート描画 ----------------
def build_period_chart(df_all, start, end, trades_df, title):
    sma5_all  = df_all["Close"].rolling(5,  min_periods=1).mean()
    sma25_all = df_all["Close"].rolling(25, min_periods=1).mean()
    df   = df_all.loc[start:end].copy()
    sma5 = sma5_all.loc[start:end]
    sma25= sma25_all.loc[start:end]

    x_idx = pd.DatetimeIndex(df_all.index).tz_localize(None)
    all_bd = pd.date_range(start=x_idx.min(), end=x_idx.max(), freq="B")
    holidays_like = all_bd.difference(x_idx.normalize())

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # 価格＆MA
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing_line_color="red", decreasing_line_color="gray",
        increasing_line_width=1.0, decreasing_line_width=1.0
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma5,  mode="lines", name="MA(5)"),  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=sma25, mode="lines", name="MA(25)"), row=1, col=1)

    # —— 売買マーカー（ここが return の“前”）——
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        # 列名ゆらぎ対応
        cols = {c.lower(): c for c in trades_df.columns}
        et_col = next((cols[c] for c in cols if c.startswith("entrytime")), None)
        ep_col = next((cols[c] for c in cols if c.startswith("entryprice")), None)
        xt_col = next((cols[c] for c in cols if c.startswith("exittime")), None)
        xp_col = next((cols[c] for c in cols if c.startswith("exitprice")), None)

        # 時刻を tz-naive に統一して期間で絞る
        def _to_naive(s):
            s = pd.to_datetime(s, errors="coerce")
            try:
                return s.dt.tz_localize(None)
            except Exception:
                return s

        if et_col and ep_col:
            e = trades_df[[et_col, ep_col]].dropna()
            e[et_col] = _to_naive(e[et_col])
            e = e[e[et_col].between(start, end, inclusive="both")]
            if not e.empty:
                fig.add_trace(go.Scatter(
                    x=e[et_col], y=pd.to_numeric(e[ep_col], errors="coerce"),
                    mode="markers", name="Buy（GC）",
                    marker=dict(symbol="triangle-up", size=14, line=dict(color="white", width=2)),
                    hovertemplate="Buy: %{x|%Y-%m-%d} @ %{y:.2f}<extra></extra>"
                ), row=1, col=1)

        if xt_col and xp_col:
            x_ = trades_df[[xt_col, xp_col]].dropna()
            x_[xt_col] = _to_naive(x_[xt_col])
            x_ = x_[x_[xt_col].between(start, end, inclusive="both")]
            if not x_.empty:
                fig.add_trace(go.Scatter(
                    x=x_[xt_col], y=pd.to_numeric(x_[xp_col], errors="coerce"),
                    mode="markers", name="Sell（DC）",
                    marker=dict(symbol="triangle-down", size=14, line=dict(color="white", width=2)),
                    hovertemplate="Sell: %{x|%Y-%m-%d} @ %{y:.2f}<extra></extra>"
                ), row=1, col=1)

    # x軸（月初のみ表示、YYYY/MM/DD）
        # --- X軸（月初だけ表示, YYYY/MM/DD） ---
        # --- X軸：各月初を表示（1日が休みでも必ず表示） ---
    # 月初の理想ラベル日
    month_starts = pd.date_range(
        start=start.normalize().replace(day=1),
        end=end.normalize(),
        freq="MS"
    )

    # 可視データのインデックス（実在する日付）
    vis_idx = pd.DatetimeIndex(df.index).tz_localize(None)

    # 月初(理想) → その月で最初に存在する営業日(実際のtick位置) へバックフィル
    tickvals, ticktext = [], []
    seen = set()
    for d in month_starts:
        pos = vis_idx.get_indexer([d], method="backfill")[0]  # d以降で最初に存在
        if pos == -1:
            continue  # 期間末などで見つからない場合はスキップ
        tv = vis_idx[pos]
        if tv in seen:
            continue     # 同じ営業日に重複しないように
        seen.add(tv)
        tickvals.append(tv)
        ticktext.append(d.strftime("%Y/%m/01"))  # ラベルは必ず「YYYY/MM/01」

    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),          # 土日を非表示
            dict(values=holidays_like.to_list())  # 祝日等も非表示
        ],
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=-45
    )



    fig.update_layout(
        title=title, xaxis_rangeslider_visible=False, hovermode="x unified",
        autosize=True, height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40), bargap=0
    )
    return fig
# ---------------- バックテスト実行 ----------------
def run_simulation(user_symbol: str, init_cash: float, commission_pct: float, months: int):
    t = normalize_ticker(user_symbol)
    company = fetch_company_name(t)
    yf_period = choose_yf_period(months)
    df_all = download_ohlcv(t, period=yf_period)

    latest = df_all.index.max()
    start  = (latest - pd.DateOffset(months=int(months))).normalize()
    df_win = df_all.loc[start:latest].dropna(how="any")
    if len(df_win) < 40:
        raise ValueError(f"期間内データが不足（{len(df_win)}本）。期間を延ばしてください。")

    lot_round = 100 if t.endswith(".T") else 1

    bt = Backtest(
        df_win,
        GoldenDeadCrossStrategy,
        cash=float(init_cash),
        commission=float(commission_pct) / 100.0,  # 片道%
        trade_on_close=False,
        exclusive_orders=True
    )
    stats = bt.run(lot_round=lot_round)

    # Final Equity 推定
    final_equity = None
    for k in stats.keys():
        if "Equity Final" in k:
            final_equity = float(stats[k]); break
    if final_equity is None:
        final_equity = float(init_cash) * (1.0 + float(stats.get("Return [%]", 0))/100.0)

    trades_count = int(stats.get("# Trades", stats.get("Trades", 0)))
    pct_return = (final_equity - float(init_cash)) / float(init_cash) * 100.0

    # trades 取得
    trades_df = None
    if hasattr(stats, "_trades") and isinstance(stats._trades, pd.DataFrame):
        trades_df = stats._trades.copy()
    elif "_trades" in stats and isinstance(stats["_trades"], pd.DataFrame):
        trades_df = stats["_trades"].copy()

    title = f"{company}（入力: {user_symbol} / 内部: {t}）— 指定期間チャート（MA5/25 & 売買ポイント）"
    fig = build_period_chart(df_all, start, latest, trades_df, title)

    meta = {
        "ticker_input": user_symbol,
        "ticker_internal": t,
        "company": company,
        "start": start.date(),
        "end": latest.date(),
        "init_cash": float(init_cash),
        "commission_pct": float(commission_pct),
        "final_equity": float(final_equity),
        "trades_count": int(trades_count),
        "pct_return": float(pct_return),
    }
    return fig, stats, trades_df, meta

# ---------------- 実行と表示 ----------------
if run_bt:
    try:
        with st.spinner("シミュレーション実行中…"):
            fig, stats, trades_df, meta = run_simulation(symbol_in, init_cash, commission_pct, months)

        # 結果サマリ
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("期間", f"{meta['start']} 〜 {meta['end']}")
        c2.metric("最終資本金(円)", f"{meta['final_equity']:,.0f}")
        c3.metric("売買回数", f"{meta['trades_count']}")
        c4.metric("リターン(%)", f"{meta['pct_return']:.2f}%")

        st.caption(f"銘柄: **{meta['company']}**（入力: `{meta['ticker_input']}`, 内部: `{meta['ticker_internal']}`）｜初期資金: {meta['init_cash']:,.0f} 円｜手数料: {meta['commission_pct']:.2f}%")

        # チャート
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # 詳細統計
        with st.expander("詳細統計（backtesting.py の結果）"):
            try:
                # 優先: to_series() があれば使う（Backtesting 公式の Stats）
                if hasattr(stats, "to_series"):
                    ser = stats.to_series()
                    df_stats = ser.to_frame(name="value")
                elif isinstance(stats, pd.Series):
                    df_stats = stats.to_frame(name="value")
                elif isinstance(stats, dict):
                    df_stats = pd.DataFrame.from_dict(stats, orient="index", columns=["value"])
                elif isinstance(stats, pd.DataFrame):
                    df_stats = stats
                else:
                    # 最後の手段：そのまま DataFrame 化
                    df_stats = pd.DataFrame(stats)

                st.dataframe(df_stats, use_container_width=True)
            except Exception as e:
                st.exception(e)


        # 取引履歴
        with st.expander("トレード履歴（エントリ/エグジット）"):
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("トレードはありませんでした。")

    except Exception as e:
        st.error(f"エラー: {e}")
        st.stop()
else:
    st.info("サイドバーで条件を設定し、**「シミュレーション実行」**を押してください。")

