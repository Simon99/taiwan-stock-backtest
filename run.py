"""Compare Original vs Pack A vs Pack B on 4 tickers."""
from strategy import run
import warnings
warnings.filterwarnings('ignore')

UNIVERSE = [
    ('0050.TW', 'TW'),
    ('2330.TW', 'TW'),
    ('SPY',     'US'),
    ('NVDA',    'US'),
]
START = '2020-01-01'
INIT = 100_000

print(f'{"ticker":<10} {"variant":<10} {"trades":>6} {"strat%":>8} {"BH%":>8} {"alpha%":>8} {"DD%":>6} {"Sharpe":>7} {"win%":>6}')
print('─' * 88)
for ticker, market in UNIVERSE:
    for variant in ['original', 'pack_a', 'pack_b', 'pack_c', 'pack_d']:
        try:
            r = run(ticker, variant, START, None, market, INIT)
            if 'error' in r:
                print(f'{ticker:<10} {variant:<10} ERROR: {r["error"]}')
                continue
            print(f'{r["ticker"]:<10} {r["variant"]:<10} {r["total_trades"]:>6} '
                  f'{r["strat_ret"]:>7.1f}% {r["bh_ret"]:>7.1f}% {r["alpha"]:>+7.1f}% '
                  f'{r["max_dd"]:>5.1f}% {r["sharpe"]:>7.2f} {r["win_rate"]:>5.1f}%')
        except Exception as e:
            import traceback
            print(f'{ticker:<10} {variant:<10} CRASH: {e}')
            traceback.print_exc()
    print()
