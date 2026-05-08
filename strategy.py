"""
均線多頭排列 + MACD 動量策略 — 三個版本：
  Original: entry=多頭排列+MACD金叉同日；exit=收盤<20MA OR trailing 10%
  Pack A:   entry=同 Original；exit=連 2 日<20MA OR ATR chandelier trailing (2.5×ATR)
  Pack B:   entry=多頭排列+DIF>signal+DIF 上升；exit=同 Pack A
"""
import vectorbt as vbt
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


def fetch_data(ticker: str, start: str = '2018-01-01', end: str | None = None,
               refresh: bool = False, max_age_hours: int = 18) -> pd.DataFrame:
    """從本地 parquet 讀；不存在或過時就重抓。
    max_age_hours: cache 超過 N 小時視為過時 (預設 18 小時，跨日後就會更新)
    """
    cache = DATA_DIR / f'{ticker.replace("/", "_")}.parquet'
    use_cache = False
    if cache.exists() and not refresh:
        age_h = (dt.datetime.now().timestamp() - cache.stat().st_mtime) / 3600
        if age_h < max_age_hours:
            use_cache = True

    if use_cache:
        df = pd.read_parquet(cache)
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f'no data for {ticker}')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        df.to_parquet(cache)

    # 即使讀 cache，仍要按使用者要求的 start/end 切片
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def cache_stats() -> dict:
    files = list(DATA_DIR.glob('*.parquet'))
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        'files': len(files),
        'total_mb': round(total_bytes / 1024 / 1024, 2),
        'tickers': sorted(f.stem for f in files),
    }


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def _kd(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 9, smooth: int = 3):
    """台股慣用 KD (9, 3, 3)。RSV → K 平滑 → D 再平滑。"""
    lo = low.rolling(n).min()
    hi = high.rolling(n).max()
    rsv = 100 * (close - lo) / (hi - lo).replace(0, np.nan)
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/smooth, adjust=False).mean()
    d = k.ewm(alpha=1/smooth, adjust=False).mean()
    return k, d


def _market_above_60ma(market: str, dates: pd.DatetimeIndex) -> pd.Series:
    """回傳大盤 (0050 / SPY) 是否站上 60MA 的布林序列，aligned 到 dates。"""
    bench = '0050.TW' if market == 'TW' else 'SPY'
    bdf = fetch_data(bench, start='2014-01-01')
    bma = bdf['Close'].rolling(60).mean()
    above = bdf['Close'] > bma
    return above.reindex(dates).ffill().fillna(False)


# ──────────────────────────────────────────
# Indicators
# ──────────────────────────────────────────
def _indicators(df: pd.DataFrame):
    close, high, low = df['Close'], df['High'], df['Low']
    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()  # 月線
    ma60 = close.rolling(60).mean()
    bull = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)

    macd = vbt.MACD.run(close, fast_window=12, slow_window=26, signal_window=9)
    dif = macd.macd
    sig = macd.signal
    if hasattr(dif, 'iloc') and dif.ndim > 1:
        dif = dif.iloc[:, 0]
        sig = sig.iloc[:, 0]
    golden_cross = (dif > sig) & (dif.shift(1) <= sig.shift(1))
    dif_rising_above = (dif > sig) & (dif > dif.shift(1))

    atr = vbt.ATR.run(high, low, close, window=14).atr
    if hasattr(atr, 'iloc') and atr.ndim > 1:
        atr = atr.iloc[:, 0]

    return {
        'close': close, 'ma20': ma20, 'atr': atr,
        'bull': bull, 'golden_cross': golden_cross, 'dif_rising_above': dif_rising_above,
    }


# ──────────────────────────────────────────
# ATR chandelier trailing exit (event-driven)
#   每進場後追蹤 highest close，當 close < highest - n_atr×ATR(today) 出場
# ──────────────────────────────────────────
def _chandelier_exits(close: pd.Series, atr: pd.Series, entries: pd.Series, n_atr: float = 2.5) -> pd.Series:
    exits = np.zeros(len(close), dtype=bool)
    in_pos = False
    high_since = -np.inf
    e_arr = entries.values
    c_arr = close.values
    a_arr = atr.values
    for i in range(len(close)):
        if e_arr[i] and not in_pos:
            in_pos = True
            high_since = c_arr[i]
            continue  # 進場當日不檢查
        if in_pos:
            if c_arr[i] > high_since:
                high_since = c_arr[i]
            if not np.isnan(a_arr[i]):
                stop = high_since - n_atr * a_arr[i]
                if c_arr[i] < stop:
                    exits[i] = True
                    in_pos = False
                    high_since = -np.inf
    return pd.Series(exits, index=close.index)


# ──────────────────────────────────────────
# Three signal variants
# ──────────────────────────────────────────
def build_original(df: pd.DataFrame):
    ind = _indicators(df)
    entries = ind['bull'] & ind['golden_cross']
    exits = (ind['close'] < ind['ma20'])  # 收盤跌破月線立即出
    return entries, exits, None  # Original 用 vbt 內建 trailing

def build_pack_a(df: pd.DataFrame, n_atr: float = 2.5):
    ind = _indicators(df)
    entries = ind['bull'] & ind['golden_cross']
    # 連 2 日跌破 20MA
    break20_2d = (ind['close'] < ind['ma20']) & (ind['close'].shift(1) < ind['ma20'].shift(1))
    chandelier = _chandelier_exits(ind['close'], ind['atr'], entries, n_atr)
    exits = break20_2d | chandelier
    return entries, exits, 'no-builtin-trail'

def build_pack_b(df: pd.DataFrame, n_atr: float = 2.5):
    ind = _indicators(df)
    entries = ind['bull'] & ind['dif_rising_above']
    break20_2d = (ind['close'] < ind['ma20']) & (ind['close'].shift(1) < ind['ma20'].shift(1))
    chandelier = _chandelier_exits(ind['close'], ind['atr'], entries, n_atr)
    exits = break20_2d | chandelier
    return entries, exits, 'no-builtin-trail'


def build_pack_c(df: pd.DataFrame, market: str = 'TW', n_atr: float = 2.5):
    """Pack B + 大盤 trend filter（0050 或 SPY 站上 60MA 才進場）"""
    ind = _indicators(df)
    market_filter = _market_above_60ma(market, df.index)
    entries = ind['bull'] & ind['dif_rising_above'] & market_filter
    break20_2d = (ind['close'] < ind['ma20']) & (ind['close'].shift(1) < ind['ma20'].shift(1))
    chandelier = _chandelier_exits(ind['close'], ind['atr'], entries, n_atr)
    exits = break20_2d | chandelier
    return entries, exits, 'no-builtin-trail'


def build_pack_e(df: pd.DataFrame):
    """Buy-the-Dip 拉回買：離 90 日高點 ≥15% + 60MA 仍上升 + 收盤 > 200MA。
    出場：連 2 日跌破 200MA（少賣，重 hold）。
    """
    close = df['Close']
    rolling_high_90 = close.rolling(90).max()
    drawdown = close / rolling_high_90 - 1
    ma60 = close.rolling(60).mean()
    ma200 = close.rolling(200).mean()

    entries = (drawdown <= -0.15) & (ma60 > ma60.shift(20)) & (close > ma200)
    exits = (close < ma200) & (close.shift(1) < ma200.shift(1))
    return entries, exits, 'no-builtin-trail'


def build_pack_f(df: pd.DataFrame):
    """RSI 超賣 + 趨勢過濾：RSI(14) 首次跌破 35 + 收盤 > 200MA 才進。
    出場：RSI > 70 OR 連 2 日跌破 200MA。
    """
    close = df['Close']
    ma200 = close.rolling(200).mean()
    rsi = vbt.RSI.run(close, window=14).rsi
    if hasattr(rsi, 'iloc') and rsi.ndim > 1:
        rsi = rsi.iloc[:, 0]

    rsi_oversold = rsi < 35
    rsi_first_oversold = rsi_oversold & ~rsi_oversold.shift(1, fill_value=False)
    above_200ma = close > ma200
    entries = rsi_first_oversold & above_200ma

    rsi_overbought = rsi > 70
    break_200ma = (close < ma200) & (close.shift(1) < ma200.shift(1))
    exits = rsi_overbought | break_200ma
    return entries, exits, 'no-builtin-trail'


def build_pack_g(df: pd.DataFrame):
    """DCA 定期定額：每月第一交易日買、跌破 200MA 才出（會 re-entry）。
    特殊處理 — run() 用 accumulate='addonly' + 動態 size。
    """
    close = df['Close']
    months = close.index.to_period('M')
    is_month_start = pd.Series(False, index=close.index)
    is_month_start.iloc[0] = True  # 第一筆當作起點
    is_month_start[months != months.shift(1).fillna(months[0])] = True

    ma200 = close.rolling(200).mean()
    exits = (close < ma200) & (close.shift(1) < ma200.shift(1))
    return is_month_start, exits, 'dca'


def build_pack_d(df: pd.DataFrame, n_atr: float = 2.5):
    """純技術指標（KD + RSI）— 不用均線排列、不用 MACD。
    Entry: KD 金叉 + K<80 + RSI>50
    Exit:  KD 死叉  OR  RSI<40  OR  ATR chandelier trailing
    """
    close, high, low = df['Close'], df['High'], df['Low']
    k, d = _kd(high, low, close, n=9, smooth=3)
    kd_golden = (k > d) & (k.shift(1) <= d.shift(1))
    kd_death  = (k < d) & (k.shift(1) >= d.shift(1))
    k_not_overbought = (k < 80)

    rsi = vbt.RSI.run(close, window=14).rsi
    if hasattr(rsi, 'iloc') and rsi.ndim > 1:
        rsi = rsi.iloc[:, 0]

    atr = vbt.ATR.run(high, low, close, window=14).atr
    if hasattr(atr, 'iloc') and atr.ndim > 1:
        atr = atr.iloc[:, 0]

    entries = kd_golden & k_not_overbought & (rsi > 50)
    chandelier = _chandelier_exits(close, atr, entries, n_atr)
    exits = kd_death | (rsi < 40) | chandelier
    return entries, exits, 'no-builtin-trail'


# ──────────────────────────────────────────
# Run backtest
# ──────────────────────────────────────────
def run(ticker: str, variant: str, start: str = '2020-01-01', end: str | None = None,
        market: str = 'TW', init_cash: float = 100_000) -> dict:
    df = fetch_data(ticker, start, end)
    if len(df) < 100:
        return {'ticker': ticker, 'error': 'too few bars'}

    if variant == 'original':
        entries, exits, _ = build_original(df)
        sl_stop, sl_trail = 0.10, True
    elif variant == 'pack_a':
        entries, exits, _ = build_pack_a(df)
        sl_stop, sl_trail = None, False
    elif variant == 'pack_b':
        entries, exits, _ = build_pack_b(df)
        sl_stop, sl_trail = None, False
    elif variant == 'pack_c':
        entries, exits, _ = build_pack_c(df, market)
        sl_stop, sl_trail = None, False
    elif variant == 'pack_d':
        entries, exits, _ = build_pack_d(df)
        sl_stop, sl_trail = None, False
    elif variant == 'pack_e':
        entries, exits, _ = build_pack_e(df)
        sl_stop, sl_trail = None, False
    elif variant == 'pack_f':
        entries, exits, _ = build_pack_f(df)
        sl_stop, sl_trail = None, False
    elif variant == 'pack_g':
        entries, exits, tag = build_pack_g(df)
        sl_stop, sl_trail = None, False
    else:
        raise ValueError(f'unknown variant {variant}')

    if market == 'TW':
        fees = 0.001425 * 0.585 + 0.003 / 2
        slippage = 0.001
    else:
        fees = 0.0
        slippage = 0.0005

    kwargs = dict(
        close=df['Close'], entries=entries, exits=exits,
        fees=fees, slippage=slippage, init_cash=init_cash, freq='1D',
    )
    if sl_stop is not None:
        kwargs['sl_stop'] = sl_stop
        kwargs['sl_trail'] = sl_trail

    # Pack G (DCA) 特殊：accumulate=addonly + 動態 size = init_cash / n_entries
    if variant == 'pack_g':
        n_e = int(entries.sum())
        if n_e == 0:
            return {'ticker': ticker, 'error': 'no DCA entry'}
        kwargs['accumulate'] = 'addonly'
        kwargs['size'] = init_cash / n_e
        kwargs['size_type'] = 'value'

    pf = vbt.Portfolio.from_signals(**kwargs)

    stats = pf.stats()
    bh = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    return {
        'ticker': ticker, 'variant': variant,
        'entry_signals': int(entries.sum()),
        'total_trades': int(stats['Total Trades']),
        'strat_ret': float(stats['Total Return [%]']),
        'bh_ret': float(bh),
        'alpha': float(stats['Total Return [%]']) - float(bh),
        'max_dd': float(stats['Max Drawdown [%]']),
        'sharpe': float(stats['Sharpe Ratio']),
        'win_rate': float(stats.get('Win Rate [%]', 0)),
        'pf': pf,
    }
