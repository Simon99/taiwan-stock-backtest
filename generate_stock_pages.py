"""
4 個投資方案 × 每方案 Top 30 列表 + 每檔股票詳細頁
  詳細頁含：4 方案排名 / 2 年日線 + MA / 新聞 / 月營收 / LLM 財報分析
"""
import json
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import warnings
import pandas as pd
import plotly.graph_objects as go
from strategy import fetch_data
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
CSV = ROOT / 'electronics_7packs.csv'
FIN = Path('/tmp/fin.json')
DOCS = ROOT / 'docs'
RECS = DOCS / 'recommendations'
STOCKS = DOCS / 'stock_pages'
RECS.mkdir(parents=True, exist_ok=True)
STOCKS.mkdir(parents=True, exist_ok=True)

GROQ_KEY = None  # loaded later
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL = 'meta-llama/llama-4-scout-17b-16e-instruct'

PACK_LABELS = {
    'pack_a': 'A 趨勢嚴格',
    'pack_b': 'B 趨勢早入',
    'pack_c': 'C 趨勢+大盤filter',
    'pack_d': 'D 純 KD/RSI',
    'pack_e': 'E 拉回 15% 買',
    'pack_f': 'F RSI 超賣',
    'pack_g': 'G DCA 月買',
}

SCHEME_NAMES = {
    1: 'Momentum 追勢',
    2: 'Steady 穩穩賺',
    3: 'Regime Adaptive',
    4: 'Future Signal',
}


def load_groq_key() -> str:
    try:
        yaml = open('/Users/chujulung/Documents/worldmonitor/docker-compose.override.yml').read()
        m = re.search(r'GROQ_API_KEY:\s*["\']?([^"\'\s#]+)', yaml)
        if m:
            return m.group(1)
    except Exception:
        pass
    return ''


def load_data():
    df = pd.read_csv(CSV, dtype={'ticker': str})
    fin = json.load(open(FIN))
    industry = {r['stock_id']: (r['stock_name'], r['industry_category'])
                for r in fin['data'] if r.get('type') == 'twse'}
    df['name'] = df['ticker'].map(lambda t: industry.get(t, (t, ''))[0])
    df['industry'] = df['ticker'].map(lambda t: industry.get(t, (t, 'Unknown'))[1])
    return df, industry


# ──────────────────────────────────────────────────────────
# Scheme scoring
# ──────────────────────────────────────────────────────────
def best_pack_per_ticker(df: pd.DataFrame, scope: str, exclude_g: bool = False) -> pd.DataFrame:
    sub = df[df['scope'] == scope].copy()
    sub = sub[(sub['trades'] > 0) & (sub['sharpe'].abs() < 100)]
    if exclude_g:
        sub = sub[sub['pack'] != 'pack_g']
    if sub.empty:
        return sub
    idx = sub.groupby('ticker')['sharpe'].idxmax()
    return sub.loc[idx].set_index('ticker')


def lookup_pack(df: pd.DataFrame, ticker: str, pack: str, scope: str) -> dict | None:
    rows = df[(df['ticker'] == ticker) & (df['pack'] == pack) & (df['scope'] == scope)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def scheme_1_momentum(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """追勢：S5 BH > 50% + total Sharpe > 1 + DD < 30%
    Score = S5_BH×0.6 + S5_strat×0.3 + (1 - DD/50)×100×0.1
    """
    total = best_pack_per_ticker(df, 'total', exclude_g=True)
    rows = []
    for t, r in total.iterrows():
        if r['sharpe'] < 1 or r['max_dd_pct'] > 30:
            continue
        s5 = lookup_pack(df, t, r['pack'], 'S5_ai')
        if not s5 or s5['bh_pct'] < 50:
            continue
        score = s5['bh_pct'] * 0.6 + s5['strat_pct'] * 0.3 + (1 - r['max_dd_pct']/50) * 100 * 0.1
        rows.append({
            'ticker': t, 'name': r['name'], 'industry': r['industry'],
            'pack': r['pack'], 'score': round(score, 2),
            'total_strat': r['strat_pct'], 'total_bh': r['bh_pct'],
            'total_sharpe': r['sharpe'], 'total_dd': r['max_dd_pct'],
            's5_bh': s5['bh_pct'], 's5_strat': s5['strat_pct'], 's5_sharpe': s5['sharpe'],
        })
    return pd.DataFrame(rows).sort_values('score', ascending=False).head(top_n).reset_index(drop=True)


def scheme_2_steady(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """穩穩賺：5 段中 ≥ 4 段正報酬 + total Sharpe > 1.2 + DD < 25%
    Score = total_Sharpe×0.5 + (5 - segment_std/30)×0.3 + S5×0.2
    """
    total = best_pack_per_ticker(df, 'total', exclude_g=True)
    rows = []
    seg_scopes = ['S1_trump1', 'S2_late', 'S3_covid', 'S4_hike', 'S5_ai']
    for t, r in total.iterrows():
        if r['sharpe'] < 1.2 or r['max_dd_pct'] > 25:
            continue
        seg_returns = []
        for s in seg_scopes:
            sg = lookup_pack(df, t, r['pack'], s)
            if sg and sg['trades'] > 0:
                seg_returns.append(sg['strat_pct'])
        if len(seg_returns) < 5:
            continue
        n_pos = sum(1 for x in seg_returns if x > 0)
        if n_pos < 4:
            continue
        seg_std = pd.Series(seg_returns).std()
        s5 = lookup_pack(df, t, r['pack'], 'S5_ai')
        s5_strat = s5['strat_pct'] if s5 else 0
        score = r['sharpe'] * 0.5 + max(0, (5 - seg_std/30)) * 0.3 + s5_strat/100 * 0.2
        rows.append({
            'ticker': t, 'name': r['name'], 'industry': r['industry'],
            'pack': r['pack'], 'score': round(score, 3),
            'total_strat': r['strat_pct'], 'total_sharpe': r['sharpe'], 'total_dd': r['max_dd_pct'],
            'n_pos_segments': n_pos, 'seg_std': round(seg_std, 1),
            's5_strat': s5_strat,
        })
    return pd.DataFrame(rows).sort_values('score', ascending=False).head(top_n).reset_index(drop=True)


def scheme_3_regime(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Regime Adaptive：S5 strat > 20% + S4 strat ≥ 0% + DD < 30% + pack ≠ G + trades ≥ 5
    Score = S5_strat×0.4 + S4_strat×0.2 + Sharpe×25 + (1-DD/50)×15
    """
    total = best_pack_per_ticker(df, 'total', exclude_g=True)
    rows = []
    for t, r in total.iterrows():
        if r['sharpe'] < 0.8 or r['max_dd_pct'] > 30 or r['trades'] < 5:
            continue
        s5 = lookup_pack(df, t, r['pack'], 'S5_ai')
        s4 = lookup_pack(df, t, r['pack'], 'S4_hike')
        if not s5 or not s4:
            continue
        if s5['strat_pct'] < 20 or s4['strat_pct'] < 0:
            continue
        score = s5['strat_pct'] * 0.4 + s4['strat_pct'] * 0.2 + r['sharpe'] * 25 + (1 - r['max_dd_pct']/50) * 15
        rows.append({
            'ticker': t, 'name': r['name'], 'industry': r['industry'],
            'pack': r['pack'], 'score': round(score, 2),
            'total_strat': r['strat_pct'], 'total_sharpe': r['sharpe'], 'total_dd': r['max_dd_pct'],
            's5_strat': s5['strat_pct'], 's5_sharpe': s5['sharpe'],
            's4_strat': s4['strat_pct'], 's4_sharpe': s4['sharpe'],
        })
    return pd.DataFrame(rows).sort_values('score', ascending=False).head(top_n).reset_index(drop=True)


def scheme_4_signal(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Future Signal：S5 期間 BH 報酬正 + S5 策略好 + DD < 30%
    Proxy 版本（不靠當下訊號 active 判斷，用 S5 整段 momentum 推估）
    Score = S5_strat×0.5 + S5_BH×0.2 + total_sharpe×30 + (1-DD/50)×20
    """
    total = best_pack_per_ticker(df, 'total', exclude_g=True)
    rows = []
    for t, r in total.iterrows():
        if r['sharpe'] < 0.8 or r['max_dd_pct'] > 30 or r['trades'] < 5:
            continue
        s5 = lookup_pack(df, t, r['pack'], 'S5_ai')
        if not s5 or s5['bh_pct'] < 0 or s5['strat_pct'] < 15:
            continue
        # S5 期間策略表現要 OK
        if s5['sharpe'] < 1.0:
            continue
        score = s5['strat_pct'] * 0.5 + s5['bh_pct'] * 0.2 + r['sharpe'] * 30 + (1 - r['max_dd_pct']/50) * 20
        rows.append({
            'ticker': t, 'name': r['name'], 'industry': r['industry'],
            'pack': r['pack'], 'score': round(score, 2),
            'total_strat': r['strat_pct'], 'total_sharpe': r['sharpe'], 'total_dd': r['max_dd_pct'],
            's5_strat': s5['strat_pct'], 's5_bh': s5['bh_pct'], 's5_sharpe': s5['sharpe'],
        })
    return pd.DataFrame(rows).sort_values('score', ascending=False).head(top_n).reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# Per-stock data (chart / news / revenue / LLM)
# ──────────────────────────────────────────────────────────
def render_chart(ticker: str) -> str:
    """plotly 2-year daily candlestick + MA5/20/60/120/240, return HTML div."""
    try:
        df = fetch_data(ticker, start='2024-01-01')  # ~2 years
        df = df.tail(500)
        if df.empty:
            return '<p class="mute">無資料可繪圖</p>'
    except Exception as e:
        return f'<p class="mute">圖表載入失敗：{e}</p>'

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='OHLC', increasing_line_color='#3fb950', decreasing_line_color='#f85149',
        showlegend=False,
    ))
    ma_specs = [(5, '#58a6ff'), (20, '#d29922'), (60, '#f0883e'),
                (120, '#a371f7'), (240, '#f85149')]
    for window, color in ma_specs:
        if len(df) >= window:
            ma = df['Close'].rolling(window).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ma, name=f'MA{window}',
                line=dict(color=color, width=1.3), mode='lines',
            ))
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn', div_id=f'chart_{ticker}')


def fetch_news(stock_name: str, ticker_id: str, limit: int = 12) -> list[dict]:
    q = urllib.parse.quote(f'{ticker_id} {stock_name}')
    url = f'https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml = urllib.request.urlopen(req, timeout=15).read()
        root = ET.fromstring(xml)
        items = []
        for item in root.iter('item'):
            t = item.find('title')
            l = item.find('link')
            d = item.find('pubDate')
            if t is None or l is None:
                continue
            items.append({
                'title': t.text or '',
                'link': l.text or '',
                'date': (d.text if d is not None else '') or '',
            })
            if len(items) >= limit:
                break
        return items
    except Exception:
        return []


def fetch_revenue(ticker_id: str) -> list[dict]:
    """FinMind 月營收，免 token。"""
    url = f'https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMonthRevenue&data_id={ticker_id}&start_date=2023-01-01'
    try:
        data = json.loads(urllib.request.urlopen(url, timeout=15).read())
        return data.get('data', [])
    except Exception:
        return []


def llm_analysis(stock_name: str, ticker_id: str, news: list, revenue: list) -> str:
    if not GROQ_KEY:
        return '（GROQ_API_KEY 未設定，跳過 LLM 分析）'
    if not news and not revenue:
        return '（無資料可分析）'

    rev_summary = ''
    if revenue:
        rev_recent = revenue[-12:] if len(revenue) > 12 else revenue
        lines = []
        for r in rev_recent:
            month = f"{r.get('revenue_year','?')}/{r.get('revenue_month','?'):02d}" if isinstance(r.get('revenue_month'), int) else f"{r.get('revenue_year','?')}-{r.get('revenue_month','?')}"
            rev_b = (r.get('revenue', 0) or 0) / 1_000_000_000
            lines.append(f"{month}: {rev_b:.2f}B")
        rev_summary = '\n'.join(lines)

    news_summary = '\n'.join(f'- {n["title"]}' for n in news[:10])

    prompt = f"""你是台股財報分析助理。針對 {ticker_id} {stock_name}：

最近 12 個月營收（單位：十億 TWD）：
{rev_summary or '（無資料）'}

最近新聞標題（最多 10 則）：
{news_summary or '（無新聞）'}

請給出 4-6 句繁體中文觀察：
1. 月營收趨勢（YoY 估算 + 近期動能）
2. 新聞主題分類（財報 / 訂單 / 法說 / 政策 / 其他）
3. 整體目前面向是利多或利空
4. 投資人需注意的風險點

只回中文觀察，不要任何開場白或結尾客套。"""

    try:
        body = {
            'model': GROQ_MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.3,
            'max_tokens': 800,
        }
        req = urllib.request.Request(
            GROQ_URL,
            data=json.dumps(body).encode(),
            headers={
                'Authorization': f'Bearer {GROQ_KEY}',
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f'（LLM 分析失敗：{e}）'


# ──────────────────────────────────────────────────────────
# HTML rendering
# ──────────────────────────────────────────────────────────
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
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin:8px 0}}
.metric{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:10px;font-size:12px}}
.metric .v{{font-size:16px;font-weight:600;color:#58a6ff;display:block;margin-top:2px}}
.scheme-badge{{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;font-weight:600;font-size:11px;margin-right:4px}}
.s1{{background:#c2530b}}.s2{{background:#1f6feb}}.s3{{background:#3fb950}}.s4{{background:#a371f7}}
.news-item{{padding:6px 0;border-bottom:1px dashed #30363d;font-size:13px}}
.news-item a{{color:#c9d1d9;text-decoration:none}}
.news-item a:hover{{color:#58a6ff;text-decoration:underline}}
.news-date{{color:#6e7681;font-size:11px;margin-left:8px}}
.llm-block{{background:#161b22;border:1px solid #30363d;border-left:3px solid #d29922;border-radius:6px;padding:12px;font-size:13px;white-space:pre-wrap;line-height:1.7}}
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
<nav><a href="/taiwan-stock-backtest/">📊 首頁</a><a href="/taiwan-stock-backtest/cross_period.html">🌍 跨週期</a><a href="/taiwan-stock-backtest/detail.html">🔬 細部</a><a href="/taiwan-stock-backtest/recommendations.html">⭐ 舊版建議</a><a href="/taiwan-stock-backtest/recommendations/">🎯 4 方案</a></nav>
'''
HTML_TAIL = '</body></html>'


def color_cell(v, kind='ret'):
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


def render_scheme_list(scheme_id: int, df: pd.DataFrame, criteria_html: str, columns: list) -> str:
    parts = [HTML_HEAD.format(title=f'方案 {scheme_id} {SCHEME_NAMES[scheme_id]}')]
    parts.append(f'<h1>🎯 方案 {scheme_id}：{SCHEME_NAMES[scheme_id]}</h1>')
    parts.append(f'<div class="section-info">{criteria_html}</div>')
    parts.append('<table class="sortable"><thead><tr>')
    for c in columns:
        parts.append(f'<th>{c[1]}</th>')
    parts.append('</tr></thead><tbody>')
    for i, (_, r) in enumerate(df.iterrows(), 1):
        parts.append('<tr>')
        for col, label in columns:
            v = r.get(col, '')
            if col == 'rank':
                parts.append(f'<td>{i}</td>')
            elif col == 'ticker':
                parts.append(f'<td><a href="../stock_pages/{r["ticker"]}.html" style="color:#58a6ff;font-weight:600">{r["ticker"]}</a></td>')
            elif col == 'pack':
                parts.append(f'<td><span class="tag">{PACK_LABELS[r["pack"]]}</span></td>')
            elif col in ('total_strat', 'total_bh', 's5_strat', 's5_bh', 's4_strat'):
                parts.append(color_cell(v, 'ret'))
            elif col in ('total_sharpe', 's5_sharpe', 's4_sharpe'):
                parts.append(color_cell(v, 'sharpe'))
            elif col == 'total_dd':
                parts.append(color_cell(v, 'dd'))
            elif col == 'score':
                parts.append(f'<td data-sort="{v}"><b>{v:.2f}</b></td>')
            else:
                parts.append(f'<td>{v}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')
    parts.append(HTML_TAIL)
    return '\n'.join(parts)


def render_recs_index(scheme_dfs: dict[int, pd.DataFrame]) -> str:
    parts = [HTML_HEAD.format(title='4 投資方案總覽')]
    parts.append('<h1>🎯 4 投資方案總覽</h1>')
    parts.append('<div class="section-info">每個方案的篩選邏輯不同，列出各自 Top 30。點 ticker 進入個股頁。</div>')

    # 哪些股票同時上多個方案榜
    overlap = {}
    for sid, df in scheme_dfs.items():
        for t in df['ticker']:
            overlap.setdefault(t, set()).add(sid)
    multi = sorted([(t, schemes) for t, schemes in overlap.items() if len(schemes) >= 2],
                    key=lambda x: -len(x[1]))

    parts.append('<h2>🌟 同時上多個方案的標的（Top 30 重疊）</h2>')
    parts.append('<table class="sortable"><thead><tr><th>Ticker</th><th>名稱</th><th>產業</th><th>方案</th><th>個別頁</th></tr></thead><tbody>')
    for t, schemes in multi:
        # 找名稱/產業
        first_match = None
        for sid in schemes:
            row = scheme_dfs[sid][scheme_dfs[sid]['ticker'] == t]
            if not row.empty:
                first_match = row.iloc[0]
                break
        if first_match is None:
            continue
        badges = ''.join(f'<span class="scheme-badge s{s}">方案{s}</span>' for s in sorted(schemes))
        parts.append('<tr>'
                     f'<td><b>{t}</b></td>'
                     f'<td>{first_match["name"]}</td>'
                     f'<td>{first_match["industry"]}</td>'
                     f'<td>{badges}</td>'
                     f'<td><a href="../stock_pages/{t}.html" style="color:#58a6ff">→ 詳細頁</a></td>'
                     '</tr>')
    parts.append('</tbody></table>')

    parts.append('<h2>📋 各方案連結</h2>')
    parts.append('<div class="metric-grid">')
    for sid in [1, 2, 3, 4]:
        df = scheme_dfs[sid]
        parts.append(f'<div class="metric"><a href="scheme_{sid}.html" style="color:#58a6ff;font-weight:600">方案 {sid}：{SCHEME_NAMES[sid]}</a><span class="v">{len(df)} 檔</span></div>')
    parts.append('</div>')

    parts.append(HTML_TAIL)
    return '\n'.join(parts)


def render_stock_page(ticker: str, df_full: pd.DataFrame, scheme_dfs: dict[int, pd.DataFrame],
                       industry_map: dict) -> str:
    """個股詳細頁。"""
    name, indust = industry_map.get(ticker, (ticker, 'Unknown'))
    parts = [HTML_HEAD.format(title=f'{ticker} {name}')]
    parts.append(f'<h1>📈 {ticker} {name} <span class="mute" style="font-size:14px;font-weight:400">{indust}</span></h1>')

    # ─── 4 方案排名 ───
    parts.append('<h2>① 4 方案排名</h2>')
    parts.append('<table><thead><tr><th>方案</th><th>排名</th><th>分數</th><th>建議 Pack</th></tr></thead><tbody>')
    for sid in [1, 2, 3, 4]:
        df = scheme_dfs[sid]
        match = df[df['ticker'] == ticker]
        if match.empty:
            parts.append(f'<tr><td>方案 {sid}：{SCHEME_NAMES[sid]}</td><td class="mute">—</td><td class="mute">未入榜</td><td class="mute">—</td></tr>')
        else:
            r = match.iloc[0]
            rank = match.index[0] + 1
            parts.append('<tr>'
                         f'<td><span class="scheme-badge s{sid}">方案{sid}</span> {SCHEME_NAMES[sid]}</td>'
                         f'<td><b>#{rank}</b></td>'
                         f'<td>{r["score"]:.2f}</td>'
                         f'<td><span class="tag">{PACK_LABELS[r["pack"]]}</span></td>'
                         '</tr>')
    parts.append('</tbody></table>')

    # ─── 圖表 ───
    parts.append('<h2>② 兩年日線（含 MA5 / MA20 / MA60 / MA120 / MA240）</h2>')
    chart_html = render_chart(f'{ticker}.TW')
    parts.append(chart_html)

    # ─── 新聞 + 月營收 + LLM ───
    parts.append('<h2>③ 最近半年新聞 + 月營收 + AI 分析</h2>')
    news = fetch_news(name, ticker)
    revenue = fetch_revenue(ticker)
    llm = llm_analysis(name, ticker, news, revenue)

    parts.append('<h3>📊 月營收（最近 12 個月，單位：十億 TWD）</h3>')
    if revenue:
        rev_recent = revenue[-12:]
        parts.append('<table><thead><tr><th>月份</th><th>營收 (B)</th><th>YoY%</th><th>MoM%</th></tr></thead><tbody>')
        for i, r in enumerate(rev_recent):
            month = f"{r.get('revenue_year','?')}/{r.get('revenue_month',0):02d}"
            rev_b = (r.get('revenue', 0) or 0) / 1_000_000_000
            yoy = mom = None
            if i >= 12:
                prev_year = revenue[i-12]
                if prev_year and prev_year.get('revenue'):
                    yoy = (rev_b - prev_year['revenue']/1_000_000_000) / (prev_year['revenue']/1_000_000_000) * 100
            if i >= 1:
                prev = rev_recent[i-1]
                if prev.get('revenue'):
                    mom = (rev_b - prev['revenue']/1_000_000_000) / (prev['revenue']/1_000_000_000) * 100
            parts.append(f'<tr><td>{month}</td>'
                         f'<td>{rev_b:.2f}</td>'
                         + (color_cell(yoy, "ret") if yoy is not None else '<td class="mute">—</td>')
                         + (color_cell(mom, "ret") if mom is not None else '<td class="mute">—</td>')
                         + '</tr>')
        parts.append('</tbody></table>')
    else:
        parts.append('<p class="mute">無月營收資料（FinMind 抓不到）</p>')

    parts.append('<h3>🤖 AI 整體觀察</h3>')
    parts.append(f'<div class="llm-block">{llm}</div>')

    parts.append('<h3>📰 最近新聞（Google News，最多 12 則）</h3>')
    if news:
        for n in news:
            parts.append(f'<div class="news-item"><a href="{n["link"]}" target="_blank" rel="noopener">{n["title"]}</a><span class="news-date">{n["date"][:16]}</span></div>')
    else:
        parts.append('<p class="mute">無新聞資料</p>')

    # ─── 7 packs 全期表現（從 csv 撈） ───
    parts.append('<h2>④ 7 packs 全期表現參考</h2>')
    sub = df_full[df_full['ticker'] == ticker]
    if sub.empty:
        parts.append('<p class="mute">無回測資料</p>')
    else:
        for scope in ['total', 'S5_ai', 'S4_hike']:
            seg = sub[sub['scope'] == scope].copy()
            if seg.empty:
                continue
            seg.loc[seg['trades'] == 0, 'sharpe'] = float('-inf')
            seg = seg.sort_values('sharpe', ascending=False)
            label_map = {'total': '總期 10y', 'S5_ai': 'S5 AI+川普2.0', 'S4_hike': 'S4 升息熊市'}
            parts.append(f'<h3>{label_map[scope]}</h3>')
            parts.append('<table><thead><tr><th>Pack</th><th>策略%</th><th>BH%</th><th>Sharpe</th><th>DD%</th><th>Trades</th></tr></thead><tbody>')
            for _, r in seg.iterrows():
                parts.append('<tr>'
                             f'<td>{PACK_LABELS[r["pack"]]}</td>'
                             + color_cell(r['strat_pct'], 'ret')
                             + color_cell(r['bh_pct'], 'ret')
                             + (color_cell(r['sharpe'], 'sharpe') if r['sharpe'] != float('-inf') else '<td class="mute">—</td>')
                             + color_cell(r['max_dd_pct'], 'dd')
                             + f'<td>{int(r["trades"])}</td>'
                             '</tr>')
            parts.append('</tbody></table>')

    parts.append(HTML_TAIL)
    return '\n'.join(parts)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    global GROQ_KEY
    GROQ_KEY = load_groq_key()
    if not GROQ_KEY:
        print('[warn] GROQ_API_KEY 未取得，LLM 分析會跳過')

    df, industry = load_data()
    print(f'載入 {len(df)} 列 / {df["ticker"].nunique()} tickers')

    # 4 schemes
    print()
    print('生成 4 投資方案...')
    s1 = scheme_1_momentum(df)
    s2 = scheme_2_steady(df)
    s3 = scheme_3_regime(df)
    s4 = scheme_4_signal(df)
    scheme_dfs = {1: s1, 2: s2, 3: s3, 4: s4}
    for sid, sdf in scheme_dfs.items():
        print(f'  方案 {sid} {SCHEME_NAMES[sid]}: {len(sdf)} 檔')

    # Render scheme list pages
    cols_1 = [('rank','#'),('ticker','Ticker'),('name','名稱'),('industry','產業'),('pack','建議 Pack'),
              ('score','分數'),('total_strat','10y策略%'),('total_dd','10y DD%'),('total_sharpe','10y Sharpe'),
              ('s5_bh','S5 BH%'),('s5_strat','S5 策略%')]
    cols_2 = [('rank','#'),('ticker','Ticker'),('name','名稱'),('industry','產業'),('pack','建議 Pack'),
              ('score','分數'),('total_strat','10y策略%'),('total_dd','10y DD%'),('total_sharpe','10y Sharpe'),
              ('n_pos_segments','正報酬段數'),('seg_std','段間std')]
    cols_3 = [('rank','#'),('ticker','Ticker'),('name','名稱'),('industry','產業'),('pack','建議 Pack'),
              ('score','分數'),('total_sharpe','10y Sharpe'),('total_dd','10y DD%'),
              ('s5_strat','S5 策略%'),('s5_sharpe','S5 Sharpe'),('s4_strat','S4 策略%'),('s4_sharpe','S4 Sharpe')]
    cols_4 = [('rank','#'),('ticker','Ticker'),('name','名稱'),('industry','產業'),('pack','建議 Pack'),
              ('score','分數'),('total_sharpe','10y Sharpe'),('total_dd','10y DD%'),
              ('s5_strat','S5 策略%'),('s5_bh','S5 BH%'),('s5_sharpe','S5 Sharpe')]

    criteria = {
        1: '<b>追勢</b>：S5 BH > 50% + 總期 Sharpe > 1 + DD < 30%。<br>分數 = S5_BH×0.6 + S5_strat×0.3 + (1-DD/50)×100×0.1。<br><b>適合</b>：相信當前漲勢延續，願追熱門股。<br><b>風險</b>：可能買在山頂、追高被套。',
        2: '<b>穩穩賺</b>：5 段中至少 4 段正報酬 + 總期 Sharpe > 1.2 + DD < 25%。<br>分數 = total_Sharpe×0.5 + (5-段間std/30)×0.3 + S5×0.2。<br><b>適合</b>：保守、要求穩定的投資人。<br><b>風險</b>：報酬天花板低、可能錯過大牛股。',
        3: '<b>Regime Adaptive</b>：S5 策略 > 20% + S4 策略 ≥ 0% + DD < 30% + Pack ≠ G + trades ≥ 5。<br>分數 = S5_strat×0.4 + S4_strat×0.2 + Sharpe×25 + (1-DD/50)×15。<br><b>適合</b>：相信市場 regime 切換，要求能適應升息+多頭兩種環境。<br><b>風險</b>：篩選嚴格，可能股池太小。',
        4: '<b>Future Signal</b>：S5 期間 BH 正 + S5 策略 > 15% + S5 Sharpe > 1 + DD < 30%。<br>分數 = S5_strat×0.5 + S5_BH×0.2 + Sharpe×30 + (1-DD/50)×20。<br><b>適合</b>：當下短中期動能 + 風控兼備。<br><b>風險</b>：仰賴 S5 持續、若 regime 改變可能失準。',
    }
    cols_map = {1: cols_1, 2: cols_2, 3: cols_3, 4: cols_4}
    for sid in [1, 2, 3, 4]:
        html = render_scheme_list(sid, scheme_dfs[sid], criteria[sid], cols_map[sid])
        (RECS / f'scheme_{sid}.html').write_text(html, encoding='utf-8')
    print(f'  → recommendations/scheme_[1-4].html')

    # Index for recommendations
    (RECS / 'index.html').write_text(render_recs_index(scheme_dfs), encoding='utf-8')
    print(f'  → recommendations/index.html')

    # Per-stock pages — union of all 4 schemes
    all_tickers = sorted(set().union(*(set(sdf['ticker']) for sdf in scheme_dfs.values())))
    print()
    print(f'生成 {len(all_tickers)} 檔個股頁（含圖表+新聞+營收+LLM）...')

    def gen_one(ticker):
        try:
            html = render_stock_page(ticker, df, scheme_dfs, industry)
            (STOCKS / f'{ticker}.html').write_text(html, encoding='utf-8')
            return ticker, True
        except Exception as e:
            return ticker, f'fail: {e}'

    # Use ThreadPool — IO-bound (news fetch / Groq / FinMind)
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(gen_one, t): t for t in all_tickers}
        for fut in as_completed(futures):
            t, result = fut.result()
            if result is True:
                done += 1
            else:
                failed += 1
                print(f'  ✗ {t}: {result}')
            if done % 5 == 0:
                print(f'  進度 {done}/{len(all_tickers)} (fail={failed})', flush=True)

    print()
    print(f'━━━ 完成 ━━━')
    print(f'  個股頁: {done} 檔成功 / {failed} 失敗')
    print(f'  目錄: {STOCKS}/')


if __name__ == '__main__':
    main()
