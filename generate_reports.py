"""生成 4 種報告：
  per_ticker/*.md            每檔股票的策略表現
  cross_period.html          跨週期 Top30 + 產業適配度
  detail.html                Sharpe>1 / alpha>0 / pivot 多角度
  recommendations.html       目前適合投入 Top 30
"""
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
CSV = ROOT / 'electronics_7packs.csv'
FIN = Path('/tmp/fin.json')  # FinMind 抓的 dump
REPORTS = ROOT / 'reports'
TICKERS = REPORTS / 'per_ticker'
REPORTS.mkdir(exist_ok=True)
TICKERS.mkdir(exist_ok=True)

PACK_LABELS = {
    'pack_a': 'A 趨勢嚴格 (MA+金叉+ATR)',
    'pack_b': 'B 趨勢早入 (MA+DIF+ATR)',
    'pack_c': 'C 趨勢+大盤filter',
    'pack_d': 'D 純 KD/RSI',
    'pack_e': 'E 拉回 15% 買',
    'pack_f': 'F RSI 超賣',
    'pack_g': 'G DCA 月買',
}
SCOPE_ORDER = ['total', 'S1_trump1', 'S2_late', 'S3_covid', 'S4_hike', 'S5_ai']
SCOPE_LABELS = {
    'total': '總期 10y',
    'S1_trump1': 'S1 川普1.0+貿易戰 (2016/05~2018/12)',
    'S2_late': 'S2 末段牛市 (2019/01~2020/02)',
    'S3_covid': 'S3 COVID+QE (2020/03~2021/12)',
    'S4_hike': 'S4 升息+通膨 (2022/01~2023/10)',
    'S5_ai': 'S5 AI+川普2.0 (2023/11~至今)',
}


def load_data():
    df = pd.read_csv(CSV, dtype={'ticker': str})
    fin = json.load(open(FIN))
    industry = {r['stock_id']: (r['stock_name'], r['industry_category'])
                for r in fin['data'] if r.get('type') == 'twse'}
    df['name'] = df['ticker'].map(lambda t: industry.get(t, (t, ''))[0])
    df['industry'] = df['ticker'].map(lambda t: industry.get(t, (t, 'Unknown'))[1])
    return df


# ─── HTML helpers ───
HTML_HEAD = '''<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang TC","SF Pro Text",sans-serif;background:#0d1117;color:#e6edf3;line-height:1.55;padding:20px;max-width:1400px;margin:0 auto}}
h1{{font-size:24px;font-weight:600;border-bottom:1px solid #30363d;padding-bottom:10px;margin-bottom:16px}}
h2{{font-size:18px;margin:28px 0 12px;color:#58a6ff;border-left:3px solid #58a6ff;padding-left:10px}}
h3{{font-size:15px;margin:18px 0 8px;color:#c9d1d9}}
nav{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:13px}}
nav a{{color:#58a6ff;margin-right:14px;text-decoration:none}}
nav a:hover{{text-decoration:underline}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0;background:#161b22;width:100%}}
th{{background:#21262d;color:#c9d1d9;padding:6px 8px;text-align:right;border:1px solid #30363d;cursor:pointer;user-select:none;font-weight:600;position:sticky;top:0}}
th:first-child,td:first-child{{text-align:left}}
th:hover{{background:#2d333b}}
td{{padding:5px 8px;border:1px solid #30363d;text-align:right;font-variant-numeric:tabular-nums}}
tr:hover td{{background:#1c2128}}
.pos{{color:#3fb950}}
.neg{{color:#f85149}}
.mute{{color:#6e7681}}
.tag{{display:inline-block;padding:2px 6px;border-radius:3px;background:#21262d;font-size:11px}}
.section-info{{color:#8b949e;font-size:12px;margin-bottom:8px}}
details{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px 12px;margin:8px 0}}
details summary{{cursor:pointer;font-weight:600;font-size:14px}}
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin:8px 0}}
.metric{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px;font-size:12px}}
.metric .v{{font-size:18px;font-weight:600;color:#58a6ff;display:block;margin-top:2px}}
.heatmap td{{position:relative}}
.legend{{font-size:11px;color:#8b949e;margin-top:4px}}
</style>
<script>
function sortTable(table,col,asc){{
  const rows=Array.from(table.querySelectorAll('tbody tr'));
  rows.sort((a,b)=>{{
    let av=a.cells[col].getAttribute('data-sort')||a.cells[col].textContent;
    let bv=b.cells[col].getAttribute('data-sort')||b.cells[col].textContent;
    const an=parseFloat(av),bn=parseFloat(bv);
    if(!isNaN(an)&&!isNaN(bn))return asc?an-bn:bn-an;
    return asc?av.localeCompare(bv):bv.localeCompare(av);
  }});
  const tb=table.querySelector('tbody');rows.forEach(r=>tb.appendChild(r));
}}
document.addEventListener('DOMContentLoaded',()=>{{
  document.querySelectorAll('table.sortable').forEach(t=>{{
    t.querySelectorAll('th').forEach((th,i)=>{{
      let asc=true;
      th.addEventListener('click',()=>{{sortTable(t,i,asc);asc=!asc;}});
    }});
  }});
}});
</script>
</head><body>
<nav><a href="index.html">📊 首頁</a><a href="cross_period.html">🌍 跨週期</a><a href="detail.html">🔬 細部分析</a><a href="recommendations.html">⭐ 投資建議</a></nav>
'''
HTML_TAIL = '</body></html>'


def color_cell(v, kind='ret'):
    """Color positive green, negative red, with proper styling."""
    if pd.isna(v):
        return '<td class="mute">—</td>'
    cls = 'pos' if v > 0 else ('neg' if v < 0 else 'mute')
    if kind == 'ret':
        return f'<td class="{cls}" data-sort="{v}">{v:+.1f}%</td>'
    elif kind == 'sharpe':
        return f'<td class="{cls}" data-sort="{v}">{v:.2f}</td>'
    elif kind == 'dd':
        return f'<td class="neg" data-sort="{v}">{v:.1f}%</td>'
    return f'<td data-sort="{v}">{v}</td>'


# ─── 1. Per-ticker markdown ───
def gen_per_ticker(df: pd.DataFrame):
    print(f'生成個股 markdown ({df["ticker"].nunique()} 檔)...')
    for ticker in df['ticker'].unique():
        sub = df[df['ticker'] == ticker]
        if sub.empty:
            continue
        first = sub.iloc[0]
        lines = []
        lines.append(f'# {ticker} {first["name"]}')
        lines.append('')
        lines.append(f'**產業**：{first["industry"]}')
        lines.append('')

        # 各 scope 一個段落（過濾 inf Sharpe，那是 0 trade 的假象）
        for scope in SCOPE_ORDER:
            seg = sub[sub['scope'] == scope].copy()
            seg.loc[seg['trades'] == 0, 'sharpe'] = float('-inf')  # 0 trade 排到最後
            seg = seg.sort_values('sharpe', ascending=False)
            if seg.empty:
                continue
            lines.append(f'## {SCOPE_LABELS[scope]}')
            lines.append('')
            lines.append('| Pack | 策略% | BH% | Alpha% | Sharpe | DD% | Trades | Win% |')
            lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
            for _, r in seg.iterrows():
                lines.append(f'| {PACK_LABELS[r["pack"]]} | {r["strat_pct"]:+.1f} | {r["bh_pct"]:+.1f} | '
                             f'{r["alpha_pct"]:+.1f} | {r["sharpe"]:.2f} | {r["max_dd_pct"]:.1f} | '
                             f'{int(r["trades"])} | {r["win_rate"]:.1f} |')
            best = seg.iloc[0]
            lines.append('')
            lines.append(f'> **本段最佳 Pack**：{PACK_LABELS[best["pack"]]} (Sharpe {best["sharpe"]:.2f})')
            lines.append('')

        path = TICKERS / f'{ticker}.md'
        path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'  → {TICKERS}/ ({len(list(TICKERS.glob("*.md")))} 檔)')


# ─── 2. Cross-period HTML ───
def gen_cross_period(df: pd.DataFrame):
    print('生成跨週期報告...')
    parts = [HTML_HEAD.format(title='跨週期分析')]
    parts.append('<h1>🌍 跨週期表現分析</h1>')

    # Section A: 各段 × 各 pack 的 Top 30
    parts.append('<h2>① 各段 × 各 pack 的 Top 30 策略報酬</h2>')
    parts.append('<p class="section-info">每個 (段, pack) 組合裡，策略報酬最高的 30 檔（含產業）。</p>')
    for scope in SCOPE_ORDER:
        parts.append(f'<h3>{SCOPE_LABELS[scope]}</h3>')
        parts.append('<details><summary>展開 7 個 pack 的 Top 30</summary>')
        for pack in PACK_LABELS:
            sub = df[(df['scope'] == scope) & (df['pack'] == pack)].copy()
            if sub.empty:
                continue
            top = sub.nlargest(30, 'strat_pct')
            parts.append(f'<h3 style="margin-top:14px;color:#8b949e;font-size:13px">{PACK_LABELS[pack]}</h3>')
            parts.append('<table class="sortable"><thead><tr>'
                         '<th>Ticker</th><th>名稱</th><th>產業</th><th>策略%</th><th>BH%</th>'
                         '<th>Alpha%</th><th>Sharpe</th><th>DD%</th></tr></thead><tbody>')
            for _, r in top.iterrows():
                parts.append('<tr>'
                             f'<td>{r["ticker"]}</td>'
                             f'<td>{r["name"]}</td>'
                             f'<td>{r["industry"]}</td>'
                             + color_cell(r['strat_pct'])
                             + color_cell(r['bh_pct'])
                             + color_cell(r['alpha_pct'])
                             + color_cell(r['sharpe'], 'sharpe')
                             + color_cell(r['max_dd_pct'], 'dd')
                             + '</tr>')
            parts.append('</tbody></table>')
        parts.append('</details>')

    # Section B: 產業 × pack 適配度（用 Sharpe 中位數）
    parts.append('<h2>② 產業 × Pack 適配度（總期 Sharpe 中位數）</h2>')
    parts.append('<p class="section-info">每個產業在每個 pack 下的 Sharpe 中位數。越高代表此 pack 對該產業越契合。</p>')
    total_df = df[df['scope'] == 'total']
    pivot = total_df.pivot_table(index='industry', columns='pack', values='sharpe', aggfunc='median')
    pivot = pivot.reindex(columns=list(PACK_LABELS.keys()))
    pivot['count'] = total_df.groupby('industry').size() // len(PACK_LABELS)
    parts.append('<table class="sortable heatmap"><thead><tr><th>產業</th>')
    for pack in PACK_LABELS:
        parts.append(f'<th>{pack[5:].upper()}</th>')
    parts.append('<th>檔數</th></tr></thead><tbody>')
    for ind, row in pivot.iterrows():
        parts.append(f'<tr><td>{ind}</td>')
        for pack in PACK_LABELS:
            v = row[pack] if pack in row else None
            if pd.isna(v):
                parts.append('<td class="mute">—</td>')
            else:
                # color intensity based on value
                if v >= 1.0:
                    bg = f'rgba(63,185,80,{min(0.6, v/2)})'
                elif v >= 0:
                    bg = f'rgba(110,118,129,{0.3})'
                else:
                    bg = f'rgba(248,81,73,{min(0.5, -v/1.5)})'
                parts.append(f'<td data-sort="{v}" style="background:{bg}">{v:.2f}</td>')
        parts.append(f'<td class="mute">{int(row["count"])}</td></tr>')
    parts.append('</tbody></table>')
    parts.append('<p class="legend">綠色 = Sharpe ≥ 1（契合）；紅色 = 負（不契合）；灰色 = 中性 0~1。</p>')

    parts.append(HTML_TAIL)
    (REPORTS / 'cross_period.html').write_text('\n'.join(parts), encoding='utf-8')
    print(f'  → cross_period.html')


# ─── 3. Detail analysis HTML ───
def gen_detail(df: pd.DataFrame):
    print('生成細部分析報告...')
    parts = [HTML_HEAD.format(title='細部分析')]
    parts.append('<h1>🔬 細部分析</h1>')

    total = df[df['scope'] == 'total']

    # Pivot：各 pack 的 stat 分布
    parts.append('<h2>① Pack 表現摘要（總期 10 年）</h2>')
    summary = total.groupby('pack').agg(
        n=('ticker', 'nunique'),
        strat_mean=('strat_pct', 'mean'),
        strat_median=('strat_pct', 'median'),
        bh_median=('bh_pct', 'median'),
        alpha_mean=('alpha_pct', 'mean'),
        sharpe_median=('sharpe', 'median'),
        sharpe_mean=('sharpe', 'mean'),
        dd_median=('max_dd_pct', 'median'),
        win_rate_mean=('win_rate', 'mean'),
    ).round(2).reindex(list(PACK_LABELS.keys()))
    parts.append('<table class="sortable"><thead><tr>'
                 '<th>Pack</th><th>檔數</th><th>策略平均%</th><th>策略中位%</th>'
                 '<th>BH中位%</th><th>Alpha平均%</th><th>Sharpe中位</th>'
                 '<th>Sharpe平均</th><th>DD中位%</th><th>Win率平均%</th>'
                 '</tr></thead><tbody>')
    for pack, r in summary.iterrows():
        parts.append('<tr>'
                     f'<td>{PACK_LABELS[pack]}</td>'
                     f'<td>{int(r["n"])}</td>'
                     + color_cell(r['strat_mean'])
                     + color_cell(r['strat_median'])
                     + color_cell(r['bh_median'])
                     + color_cell(r['alpha_mean'])
                     + color_cell(r['sharpe_median'], 'sharpe')
                     + color_cell(r['sharpe_mean'], 'sharpe')
                     + color_cell(r['dd_median'], 'dd')
                     + f'<td data-sort="{r["win_rate_mean"]}">{r["win_rate_mean"]:.1f}</td>'
                     + '</tr>')
    parts.append('</tbody></table>')

    # Sharpe > 1 的所有 (ticker, pack, scope)，排除 inf 跟 0 trade
    sharpe_gt1 = df[(df['sharpe'] > 1) & (df['sharpe'] != float('inf')) & (df['trades'] > 0)].copy()
    parts.append(f'<h2>② Sharpe &gt; 1 的所有結果（共 {len(sharpe_gt1)} 筆）</h2>')
    parts.append('<p class="section-info">點欄位排序。完整列表，可篩選找適合自己風險承受度的標的。</p>')
    sharpe_gt1 = sharpe_gt1.sort_values('sharpe', ascending=False)
    parts.append('<table class="sortable"><thead><tr>'
                 '<th>Ticker</th><th>名稱</th><th>產業</th><th>Pack</th><th>Scope</th>'
                 '<th>策略%</th><th>BH%</th><th>Alpha%</th><th>Sharpe</th><th>DD%</th>'
                 '</tr></thead><tbody>')
    for _, r in sharpe_gt1.head(500).iterrows():
        parts.append('<tr>'
                     f'<td>{r["ticker"]}</td>'
                     f'<td>{r["name"]}</td>'
                     f'<td>{r["industry"]}</td>'
                     f'<td><span class="tag">{r["pack"][5:].upper()}</span></td>'
                     f'<td>{r["scope"]}</td>'
                     + color_cell(r['strat_pct'])
                     + color_cell(r['bh_pct'])
                     + color_cell(r['alpha_pct'])
                     + color_cell(r['sharpe'], 'sharpe')
                     + color_cell(r['max_dd_pct'], 'dd')
                     + '</tr>')
    parts.append('</tbody></table>')
    if len(sharpe_gt1) > 500:
        parts.append(f'<p class="mute">… 顯示前 500 筆，總計 {len(sharpe_gt1)}</p>')

    # Alpha > 0
    alpha_pos = df[df['alpha_pct'] > 0].copy()
    parts.append(f'<h2>③ Alpha &gt; 0 — 策略勝過 BH 的清單（共 {len(alpha_pos)} 筆）</h2>')
    parts.append('<p class="section-info">這些是「策略真的有用」的證據。但注意，多數出現在 BH 自己很爛的標的（策略至少不全程承受跌幅）。</p>')
    alpha_pos = alpha_pos.sort_values('alpha_pct', ascending=False)
    parts.append('<table class="sortable"><thead><tr>'
                 '<th>Ticker</th><th>名稱</th><th>產業</th><th>Pack</th><th>Scope</th>'
                 '<th>Alpha%</th><th>策略%</th><th>BH%</th><th>Sharpe</th>'
                 '</tr></thead><tbody>')
    for _, r in alpha_pos.head(500).iterrows():
        parts.append('<tr>'
                     f'<td>{r["ticker"]}</td>'
                     f'<td>{r["name"]}</td>'
                     f'<td>{r["industry"]}</td>'
                     f'<td><span class="tag">{r["pack"][5:].upper()}</span></td>'
                     f'<td>{r["scope"]}</td>'
                     + color_cell(r['alpha_pct'])
                     + color_cell(r['strat_pct'])
                     + color_cell(r['bh_pct'])
                     + color_cell(r['sharpe'], 'sharpe')
                     + '</tr>')
    parts.append('</tbody></table>')
    if len(alpha_pos) > 500:
        parts.append(f'<p class="mute">… 顯示前 500 筆，總計 {len(alpha_pos)}</p>')

    parts.append(HTML_TAIL)
    (REPORTS / 'detail.html').write_text('\n'.join(parts), encoding='utf-8')
    print('  → detail.html')


# ─── 4. Investment recommendations ───
def gen_recommendations(df: pd.DataFrame):
    print('生成投資建議報告...')
    # 為每檔股票找「最適合的 pack」（用總期 Sharpe，排除 inf / 0 trade）
    total_all = df[df['scope'] == 'total'].copy()
    total = total_all[(total_all['trades'] > 0) & (total_all['sharpe'] != float('inf')) & (total_all['sharpe'].abs() < 100)].copy()
    s5 = df[df['scope'] == 'S5_ai'].copy()
    s4 = df[df['scope'] == 'S4_hike'].copy()

    # 每檔 stock × 找最佳 pack（基於總期 Sharpe）
    best_total = total.loc[total.groupby('ticker')['sharpe'].idxmax()].set_index('ticker')

    # 對應的 S5 / S4 表現（用同一個 pack）
    rows = []
    for ticker, r in best_total.iterrows():
        pack = r['pack']
        s5_row = s5[(s5['ticker'] == ticker) & (s5['pack'] == pack)]
        s4_row = s4[(s4['ticker'] == ticker) & (s4['pack'] == pack)]
        if s5_row.empty or s4_row.empty:
            continue
        s5_sharpe = s5_row.iloc[0]['sharpe']
        s5_ret = s5_row.iloc[0]['strat_pct']
        s4_sharpe = s4_row.iloc[0]['sharpe']

        # 過濾 inf / nan
        if not all(pd.notna([r['sharpe'], s5_sharpe, s4_sharpe])):
            continue
        if any(abs(x) > 100 for x in [r['sharpe'], s5_sharpe, s4_sharpe]):  # inf 等
            continue

        # 篩選條件
        if r['sharpe'] < 0.8:
            continue
        if s5_sharpe < 1.0:
            continue
        if s4_sharpe < 0.3:
            continue

        # 加權打分
        score = s5_sharpe * 0.5 + r['sharpe'] * 0.3 + s4_sharpe * 0.2
        rows.append({
            'ticker': ticker,
            'name': r['name'],
            'industry': r['industry'],
            'best_pack': pack,
            'total_sharpe': r['sharpe'],
            'total_strat': r['strat_pct'],
            'total_bh': r['bh_pct'],
            'total_dd': r['max_dd_pct'],
            's5_sharpe': s5_sharpe,
            's5_strat': s5_ret,
            's4_sharpe': s4_sharpe,
            'score': score,
        })
    rec = pd.DataFrame(rows).sort_values('score', ascending=False).head(30)

    parts = [HTML_HEAD.format(title='投資建議')]
    parts.append('<h1>⭐ 目前適合投入的 30 檔（依綜合分數排序）</h1>')
    parts.append('<div class="section-info">')
    parts.append('<p><b>篩選條件</b>（all 須滿足）：</p>')
    parts.append('<ol style="padding-left:24px">')
    parts.append('<li>該檔最佳 pack 的<b>總期 (10y) Sharpe ≥ 0.8</b>（長期策略契合度足夠）</li>')
    parts.append('<li>同 pack 在 <b>S5 (AI+川普2.0, 近 18 個月) Sharpe ≥ 1.0</b>（當前市場環境表現好）</li>')
    parts.append('<li>同 pack 在 <b>S4 (升息熊市) Sharpe ≥ 0.3</b>（能撐熊市，下行可控）</li>')
    parts.append('</ol>')
    parts.append('<p><b>綜合分數 = S5 Sharpe × 0.5 + 總期 Sharpe × 0.3 + S4 Sharpe × 0.2</b></p>')
    parts.append('<p class="mute">注意：此為機械式回測排名，<b>不構成投資建議</b>。實盤決策須加入基本面、籌碼面、總體環境判斷。</p>')
    parts.append('</div>')

    parts.append('<table class="sortable"><thead><tr>'
                 '<th>排名</th><th>Ticker</th><th>名稱</th><th>產業</th>'
                 '<th>建議 Pack</th><th>分數</th>'
                 '<th>10y Sharpe</th><th>10y策略%</th><th>10y BH%</th><th>10y DD%</th>'
                 '<th>S5 Sharpe</th><th>S5策略%</th><th>S4 Sharpe</th>'
                 '</tr></thead><tbody>')
    for i, (_, r) in enumerate(rec.iterrows(), 1):
        parts.append('<tr>'
                     f'<td>{i}</td>'
                     f'<td><a href="per_ticker/{r["ticker"]}.md" style="color:#58a6ff">{r["ticker"]}</a></td>'
                     f'<td>{r["name"]}</td>'
                     f'<td>{r["industry"]}</td>'
                     f'<td><span class="tag">{r["best_pack"][5:].upper()}</span></td>'
                     f'<td data-sort="{r["score"]}"><b>{r["score"]:.2f}</b></td>'
                     + color_cell(r['total_sharpe'], 'sharpe')
                     + color_cell(r['total_strat'])
                     + color_cell(r['total_bh'])
                     + color_cell(r['total_dd'], 'dd')
                     + color_cell(r['s5_sharpe'], 'sharpe')
                     + color_cell(r['s5_strat'])
                     + color_cell(r['s4_sharpe'], 'sharpe')
                     + '</tr>')
    parts.append('</tbody></table>')

    # 推薦理由 — 每檔個別說明
    parts.append('<h2>每檔推薦理由</h2>')
    for i, (_, r) in enumerate(rec.iterrows(), 1):
        pack_short = r['best_pack'][5:].upper()
        why = []
        why.append(f'**最契合 Pack：{PACK_LABELS[r["best_pack"]]}**（10 年 Sharpe {r["total_sharpe"]:.2f}）')
        if r['s5_sharpe'] >= 1.5:
            why.append(f'AI+川普2.0 階段表現極佳（Sharpe {r["s5_sharpe"]:.2f}, +{r["s5_strat"]:.1f}%）')
        elif r['s5_sharpe'] >= 1.0:
            why.append(f'近期 (S5) 表現穩定（Sharpe {r["s5_sharpe"]:.2f}, +{r["s5_strat"]:.1f}%）')
        if r['s4_sharpe'] >= 1.0:
            why.append(f'升息熊市仍正報酬（S4 Sharpe {r["s4_sharpe"]:.2f}），下行控制好')
        elif r['s4_sharpe'] >= 0.5:
            why.append(f'升息熊市 Sharpe {r["s4_sharpe"]:.2f}，能撐')
        else:
            why.append(f'升息熊市 Sharpe {r["s4_sharpe"]:.2f}，需注意下行風險')
        if r['total_dd'] < 25:
            why.append(f'最大回撤僅 {r["total_dd"]:.1f}%，風險可控')
        elif r['total_dd'] < 40:
            why.append(f'最大回撤 {r["total_dd"]:.1f}%，中等')
        else:
            why.append(f'⚠️ 最大回撤 {r["total_dd"]:.1f}% 偏大')

        parts.append(f'<details><summary>#{i} {r["ticker"]} {r["name"]} ({r["industry"]}) — 分數 {r["score"]:.2f}</summary>')
        parts.append('<ul style="padding-left:20px;margin-top:8px">')
        for w in why:
            parts.append(f'<li>{w}</li>')
        parts.append('</ul></details>')

    parts.append(HTML_TAIL)
    (REPORTS / 'recommendations.html').write_text('\n'.join(parts), encoding='utf-8')
    print('  → recommendations.html')
    return rec


# ─── 5. Index ───
def gen_index(df: pd.DataFrame, rec: pd.DataFrame):
    n_ticker = df['ticker'].nunique()
    parts = [HTML_HEAD.format(title='Algo Trading Reports')]
    parts.append('<h1>📊 台股電子股策略回測報告</h1>')
    parts.append('<div class="metric-grid">')
    parts.append(f'<div class="metric">總 ticker 數<span class="v">{n_ticker}</span></div>')
    parts.append(f'<div class="metric">資料筆數<span class="v">{len(df):,}</span></div>')
    parts.append(f'<div class="metric">策略 (packs)<span class="v">7</span></div>')
    parts.append(f'<div class="metric">時段<span class="v">總 + 5 段</span></div>')
    parts.append(f'<div class="metric">推薦標的<span class="v">{len(rec)}</span></div>')
    parts.append('</div>')

    parts.append('<h2>報告導覽</h2>')
    parts.append('<ul style="padding-left:24px;line-height:2">')
    parts.append('<li><a href="cross_period.html" style="color:#58a6ff">🌍 跨週期分析</a> — 各段 Top 30 + 產業適配度</li>')
    parts.append('<li><a href="detail.html" style="color:#58a6ff">🔬 細部分析</a> — Pack 摘要 + Sharpe&gt;1 + alpha&gt;0 清單</li>')
    parts.append('<li><a href="recommendations.html" style="color:#58a6ff">⭐ 投資建議</a> — 目前適合投入 Top 30</li>')
    parts.append('<li><a href="per_ticker/" style="color:#58a6ff">📂 個股 markdown</a> — 449 檔詳細表現</li>')
    parts.append('</ul>')

    parts.append('<h2>資料說明</h2>')
    parts.append('<table><thead><tr><th>項目</th><th>內容</th></tr></thead><tbody>')
    parts.append('<tr><td>Universe</td><td>TWSE 上市電子股 540 檔（FinMind API）</td></tr>')
    parts.append('<tr><td>資料期間</td><td>2016-05-07 ~ 至今（10 年）</td></tr>')
    parts.append('<tr><td>5 個歷史段</td><td>S1 川普1.0+貿易戰 / S2 末段牛市 / S3 COVID+QE / S4 升息熊市 / S5 AI+川普2.0</td></tr>')
    parts.append('<tr><td>策略 packs</td><td>A 趨勢嚴格 / B 趨勢早入 / C 趨勢+大盤filter / D 純 KD/RSI / E 拉回買 / F RSI 超賣 / G DCA</td></tr>')
    parts.append('<tr><td>資料源</td><td>yfinance (本地 parquet cache)</td></tr>')
    parts.append('</tbody></table>')

    parts.append(HTML_TAIL)
    (REPORTS / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print('  → index.html')


def main():
    df = load_data()
    print(f'載入：{len(df)} rows / {df["ticker"].nunique()} tickers / {df["industry"].nunique()} industries')
    print()
    gen_per_ticker(df)
    gen_cross_period(df)
    gen_detail(df)
    rec = gen_recommendations(df)
    gen_index(df, rec)
    print()
    print(f'━━━ 全部完成 → {REPORTS}/ ━━━')
    print(f'  入口: open {REPORTS / "index.html"}')


if __name__ == '__main__':
    main()
