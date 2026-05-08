"""單檔股票完整分析 — 4 packs × 6 segments，內部 ThreadPool 並行。
usage: python analyze_one.py 2308.TW [--no-parallel]
"""
import sys
import time
import datetime as dt
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from strategy import run, fetch_data
warnings.filterwarnings('ignore')

TODAY = dt.date.today().strftime('%Y-%m-%d')
SEGMENTS = [
    ('total',     '2016-05-07', TODAY),
    ('S1_trump1', '2016-05-07', '2018-12-31'),
    ('S2_late',   '2019-01-01', '2020-02-29'),
    ('S3_covid',  '2020-03-01', '2021-12-31'),
    ('S4_hike',   '2022-01-01', '2023-10-31'),
    ('S5_ai',     '2023-11-01', TODAY),
]
PACKS = ['pack_a', 'pack_b', 'pack_c', 'pack_d']
INIT_CASH = 100_000


def _one_combo(args):
    ticker, pack, label, s, e = args
    try:
        r = run(ticker, pack, start=s, end=e, market='TW', init_cash=INIT_CASH)
        if 'error' in r:
            return None
        return {
            'pack': pack,
            'scope': label,
            'strat_pct': round(r['strat_ret'], 2),
            'bh_pct': round(r['bh_ret'], 2),
            'alpha_pct': round(r['alpha'], 2),
            'sharpe': round(r['sharpe'], 3),
            'max_dd_pct': round(r['max_dd'], 2),
            'trades': r['total_trades'],
            'win_rate': round(r['win_rate'], 1),
        }
    except Exception:
        return None


def analyze_sequential(ticker: str) -> list[dict]:
    rows = []
    for pack in PACKS:
        for label, s, e in SEGMENTS:
            r = _one_combo((ticker, pack, label, s, e))
            if r:
                rows.append(r)
    return rows


def analyze_parallel(ticker: str, max_workers: int = 8) -> list[dict]:
    combos = [(ticker, p, l, s, e) for p in PACKS for l, s, e in SEGMENTS]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_one_combo, combos))
    return [r for r in results if r is not None]


def main():
    if len(sys.argv) < 2:
        print('usage: python analyze_one.py 2308.TW [--no-parallel]')
        sys.exit(1)
    ticker = sys.argv[1]
    if not ticker.endswith('.TW'):
        ticker += '.TW'
    no_parallel = '--no-parallel' in sys.argv

    # Pre-fetch (warm cache + numba JIT 一次)
    print(f'分析 {ticker}...')
    fetch_data(ticker, start='2016-01-01')

    # Warmup JIT with one quick call
    _ = run(ticker, 'pack_b', start='2024-01-01', end=TODAY, market='TW', init_cash=INIT_CASH)

    # Sequential timing
    t0 = time.perf_counter()
    rows_seq = analyze_sequential(ticker)
    seq_time = time.perf_counter() - t0
    print(f'  Sequential:  {seq_time*1000:.0f} ms ({len(rows_seq)} rows)')

    if not no_parallel:
        # Parallel timing
        t0 = time.perf_counter()
        rows_par = analyze_parallel(ticker, max_workers=8)
        par_time = time.perf_counter() - t0
        speedup = seq_time / par_time if par_time > 0 else 0
        print(f'  Parallel(8): {par_time*1000:.0f} ms ({len(rows_par)} rows)  → {speedup:.1f}x speedup')

    # 顯示結果（用 sequential 結果）
    df = pd.DataFrame(rows_seq)
    if df.empty:
        print('no data')
        return
    print()
    print(f'━━━ {ticker} 完整分析 ━━━')
    pivot_strat = df.pivot(index='pack', columns='scope', values='strat_pct')
    pivot_strat = pivot_strat[['total', 'S1_trump1', 'S2_late', 'S3_covid', 'S4_hike', 'S5_ai']]
    print('\n策略報酬 %:')
    print(pivot_strat.round(1).to_string())

    pivot_bh = df.pivot(index='pack', columns='scope', values='bh_pct')
    pivot_bh = pivot_bh[['total', 'S1_trump1', 'S2_late', 'S3_covid', 'S4_hike', 'S5_ai']]
    print('\nBuy & Hold %:')
    print(pivot_bh.round(1).to_string())

    pivot_sharpe = df.pivot(index='pack', columns='scope', values='sharpe')
    pivot_sharpe = pivot_sharpe[['total', 'S1_trump1', 'S2_late', 'S3_covid', 'S4_hike', 'S5_ai']]
    print('\nSharpe:')
    print(pivot_sharpe.round(2).to_string())


if __name__ == '__main__':
    main()
