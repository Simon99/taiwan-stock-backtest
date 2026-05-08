"""
所有上市電子股 × 4 套策略 × 6 scope (total + 5 段) → 純資料 dump
並行版本：ProcessPoolExecutor 跑多核（M3 Max 預設 8 workers）
每處理 10 檔印一次進度 + 存 partial CSV。
"""
import os
import sys
import warnings
import datetime as dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from strategy import run, fetch_data, cache_stats
from tw_electronics_full import ELECTRONICS_TICKERS as ELECTRONICS
warnings.filterwarnings('ignore')

OUT = Path(__file__).parent / 'electronics_7packs.csv'
PARTIAL = Path(__file__).parent / 'electronics_7packs.partial'

TODAY = dt.date.today().strftime('%Y-%m-%d')
SEGMENTS = [
    ('total',     '2016-05-07', TODAY),
    ('S1_trump1', '2016-05-07', '2018-12-31'),
    ('S2_late',   '2019-01-01', '2020-02-29'),
    ('S3_covid',  '2020-03-01', '2021-12-31'),
    ('S4_hike',   '2022-01-01', '2023-10-31'),
    ('S5_ai',     '2023-11-01', TODAY),
]
PACKS = ['pack_a', 'pack_b', 'pack_c', 'pack_d', 'pack_e', 'pack_f', 'pack_g']
INIT_CASH = 100_000

# M3 Max 16 核 (12P + 4E)；並行 backtest 用 8 個 workers 留 headroom 給 OS / cache loading
N_WORKERS = int(os.environ.get('N_WORKERS', '8'))


def process_ticker(ticker: str) -> list[dict]:
    """Worker function — 跑單一 ticker 全部 packs × segments。
    必須為 top-level 才能被 ProcessPoolExecutor pickle。
    """
    try:
        df = fetch_data(ticker, start='2016-01-01')
        if len(df) < 200:
            return []
    except Exception:
        return []

    rows = []
    for pack in PACKS:
        for label, s, e in SEGMENTS:
            try:
                r = run(ticker, pack, start=s, end=e, market='TW', init_cash=INIT_CASH)
                if 'error' in r:
                    continue
                rows.append({
                    'ticker': ticker.replace('.TW', ''),
                    'pack': pack,
                    'scope': label,
                    'strat_pct': round(r['strat_ret'], 2),
                    'bh_pct': round(r['bh_ret'], 2),
                    'alpha_pct': round(r['alpha'], 2),
                    'sharpe': round(r['sharpe'], 3),
                    'max_dd_pct': round(r['max_dd'], 2),
                    'trades': r['total_trades'],
                    'win_rate': round(r['win_rate'], 1),
                })
            except Exception:
                pass
    return rows


def main():
    n = len(ELECTRONICS)
    n_bt = n * len(PACKS) * len(SEGMENTS)
    print(f'掃描 {n} 檔上市電子股 × 4 packs × 6 scopes = {n_bt} 個 backtest')
    print(f'並行 workers: {N_WORKERS}')
    print(f'(資料不足的段會自動 skip)')
    print()

    all_rows = []
    fail = 0
    completed = 0

    t0 = dt.datetime.now()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_ticker, t): t for t in ELECTRONICS}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                rows = fut.result()
                if not rows:
                    fail += 1
                else:
                    all_rows.extend(rows)
            except Exception as ex_:
                fail += 1
            completed += 1
            if completed % 10 == 0:
                elapsed = (dt.datetime.now() - t0).total_seconds()
                eta = elapsed / completed * (n - completed)
                df_partial = pd.DataFrame(all_rows)
                partial_file = PARTIAL.with_suffix(f'.{completed:03d}.csv')
                df_partial.to_csv(partial_file, index=False)
                print(f'  進度 {completed}/{n}  fail={fail}  rows={len(all_rows)}  '
                      f'elapsed={elapsed:.0f}s  ETA={eta:.0f}s  → {partial_file.name}', flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)
    total_elapsed = (dt.datetime.now() - t0).total_seconds()
    print()
    print(f'━━━ 完成（總耗時 {total_elapsed:.1f}s）━━━')
    print(f'  有效資料筆數: {len(df)}')
    print(f'  失敗 ticker:  {fail}')
    print(f'  存到:        {OUT.name}')
    print()
    print(f'━━━ 樣本檢查 ━━━')
    print(df.head(8).to_string(index=False))
    print()
    print(f'━━━ 各 pack × scope 平均策略報酬 ━━━')
    pivot = df.pivot_table(index='pack', columns='scope', values='strat_pct', aggfunc='mean').round(1)
    pivot = pivot[['total', 'S1_trump1', 'S2_late', 'S3_covid', 'S4_hike', 'S5_ai']]
    print(pivot.to_string())
    print()
    cs = cache_stats()
    print(f'cache: {cs["files"]} 檔 / {cs["total_mb"]} MB')


if __name__ == '__main__':
    main()
