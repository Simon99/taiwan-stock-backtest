"""
1. 取廣域 TW universe (~150-200 檔)
2. 撈 10 年資料 → 算 5 年 BH 報酬 → 挑前 100 名
3. 對前 100 名跑 Pack B 10 年回測，分 5 段顯示
4. 每測完 10 檔輸出一次中途報告，最後存 CSV
"""
import sys
import warnings
import pandas as pd
import datetime as dt
from pathlib import Path
from strategy import fetch_data, run, cache_stats
from tw_universe import UNIVERSE
warnings.filterwarnings('ignore')

OUT = Path(__file__).parent / 'top100_report.csv'
PARTIAL = Path(__file__).parent / 'top100_partial.txt'

START_FETCH = '2016-01-01'
TODAY = dt.date.today().strftime('%Y-%m-%d')
START_BT = '2016-05-07'  # 10 年前（保留 4 個月 indicator warmup 從 2016-01）
START_5Y = '2021-05-07'

# 5 個歷史段（按重大事件）
SEGMENTS = [
    ('S1: 川普1.0+貿易戰',  '2016-05-07', '2018-12-31'),
    ('S2: 末段牛市',         '2019-01-01', '2020-02-29'),
    ('S3: COVID+QE暴漲',    '2020-03-01', '2021-12-31'),
    ('S4: 升息熊市+通膨',    '2022-01-01', '2023-10-31'),
    ('S5: AI+川普2.0',       '2023-11-01', TODAY),
]

INIT_CASH = 100_000


def step1_fetch_and_rank() -> list[tuple[str, float]]:
    """Fetch all, compute 5y BH return, return list of (ticker, 5y_return) sorted desc."""
    print(f'Step 1: 抓 {len(UNIVERSE)} 檔 10 年資料 + 算 5 年 BH 報酬...')
    rows = []
    fail = 0
    for i, t in enumerate(UNIVERSE):
        try:
            df = fetch_data(t, start=START_FETCH)
            if df.empty or len(df) < 200:
                fail += 1; continue
            # 5 年 BH
            df_5y = df[df.index >= pd.Timestamp(START_5Y)]
            if len(df_5y) < 100:
                fail += 1; continue
            ret = (df_5y['Close'].iloc[-1] / df_5y['Close'].iloc[0] - 1) * 100
            rows.append((t, float(ret)))
        except Exception:
            fail += 1
        if (i + 1) % 20 == 0:
            print(f'  進度 {i+1}/{len(UNIVERSE)} (fail={fail})', flush=True)
    print(f'  完成: {len(rows)} 檔有效，{fail} 檔失敗')
    rows.sort(key=lambda x: -x[1])
    return rows


def run_segmented_backtest(ticker: str) -> dict:
    """跑 Pack B 10 年回測 + 每段獨立回測，回傳結果 dict。"""
    out = {'ticker': ticker.replace('.TW', '')}
    # 全期 10 年
    full = run(ticker, 'pack_b', start=START_BT, end=TODAY, market='TW', init_cash=INIT_CASH)
    if 'error' in full:
        out['error'] = full['error']; return out
    out['total_strat%'] = round(full['strat_ret'], 1)
    out['total_BH%'] = round(full['bh_ret'], 1)
    out['total_alpha%'] = round(full['alpha'], 1)
    out['total_DD%'] = round(full['max_dd'], 1)
    out['total_Sharpe'] = round(full['sharpe'], 2)
    out['total_trades'] = full['total_trades']
    # 各段
    for label, s, e in SEGMENTS:
        try:
            r = run(ticker, 'pack_b', start=s, end=e, market='TW', init_cash=INIT_CASH)
            if 'error' in r:
                out[label + '_strat%'] = None
                out[label + '_BH%'] = None
                continue
            out[label + '_strat%'] = round(r['strat_ret'], 1)
            out[label + '_BH%'] = round(r['bh_ret'], 1)
        except Exception:
            out[label + '_strat%'] = None
            out[label + '_BH%'] = None
    return out


def step2_backtest_top100(ranked: list[tuple[str, float]]):
    """前 100 名各跑 10 年分段回測，每 10 檔輸出進度。"""
    top = ranked[:100]
    print(f'\nStep 2: 對前 {len(top)} 名跑 10 年 Pack B 回測（5 段）...\n')
    print(f'{"#":>3} {"ticker":<8} {"5y BH%":>8} {"10y strat":>10} {"10y BH":>8} {"DD":>6} {"Sharpe":>7}  ' + '  '.join([f'{l.split(":")[0]+"%":>9}' for l, _, _ in SEGMENTS]))
    print('─' * 130)
    results = []
    for i, (t, bh5y) in enumerate(top, 1):
        try:
            r = run_segmented_backtest(t)
            r['rank'] = i
            r['5y_BH%'] = round(bh5y, 1)
            results.append(r)
            if 'error' in r:
                print(f'{i:>3} {r["ticker"]:<8} ERROR: {r["error"]}')
            else:
                seg_strs = []
                for label, _, _ in SEGMENTS:
                    s = r.get(label + '_strat%')
                    seg_strs.append(f'{s:>+8.1f}%' if s is not None else f'{"--":>9}')
                print(f'{i:>3} {r["ticker"]:<8} {bh5y:>+7.0f}% '
                      f'{r["total_strat%"]:>+9.0f}% {r["total_BH%"]:>+7.0f}% '
                      f'{r["total_DD%"]:>5.1f}% {r["total_Sharpe"]:>6.2f}   '
                      + '  '.join(seg_strs))
        except Exception as e:
            print(f'{i:>3} {t} CRASH: {str(e)[:80]}')
        # 每 10 檔輸出 partial 報告
        if i % 10 == 0:
            df_partial = pd.DataFrame([r for r in results if 'error' not in r])
            df_partial.to_csv(PARTIAL.with_suffix(f'.{i:03d}.csv'), index=False)
            print(f'  [partial saved → {PARTIAL.stem}.{i:03d}.csv，已測 {i}/{len(top)} 檔]', flush=True)
    return results


if __name__ == '__main__':
    ranked = step1_fetch_and_rank()
    if not ranked:
        sys.exit('No valid tickers')
    print(f'\n5 年 BH 報酬前 10 名：')
    for t, r in ranked[:10]:
        print(f'  {t.replace(".TW",""):<8} {r:>+10.1f}%')

    results = step2_backtest_top100(ranked)
    valid = [r for r in results if 'error' not in r]
    df = pd.DataFrame(valid)
    df.to_csv(OUT, index=False)
    print(f'\n━━━ 全部完成。結果存到 {OUT.name} ({len(df)} 檔有效)━━━')

    # 摘要
    df_sorted = df.sort_values('total_Sharpe', ascending=False)
    print('\n總 Sharpe 前 15 名:')
    cols = ['rank', 'ticker', 'total_strat%', 'total_BH%', 'total_alpha%', 'total_DD%', 'total_Sharpe', 'total_trades']
    print(df_sorted[cols].head(15).to_string(index=False))

    print('\n各段 Pack B 平均策略報酬（看哪段跑得最好）:')
    for label, _, _ in SEGMENTS:
        col = label + '_strat%'
        if col in df.columns:
            avg = df[col].dropna().mean()
            print(f'  {label}: 平均 {avg:+.1f}%')

    cs = cache_stats()
    print(f'\ncache: {cs["files"]} 檔 / {cs["total_mb"]} MB')
