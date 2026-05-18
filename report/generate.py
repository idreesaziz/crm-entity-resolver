from __future__ import annotations

import datetime as dt
import html
from pathlib import Path
from typing import Any


def _badge_class(conf: float) -> str:
    if conf >= 0.95:
        return "badge badge--green"
    if conf >= 0.85:
        return "badge badge--yellow"
    return "badge badge--gray"


def _confidence_buckets(confs: list[float]) -> list[tuple[str, int]]:
    buckets = [
        ("0.85–0.89", 0),
        ("0.90–0.94", 0),
        ("0.95–0.97", 0),
        ("0.98–1.00", 0),
    ]
    for c in confs:
        if 0.85 <= c < 0.90:
            buckets[0] = (buckets[0][0], buckets[0][1] + 1)
        elif 0.90 <= c < 0.95:
            buckets[1] = (buckets[1][0], buckets[1][1] + 1)
        elif 0.95 <= c < 0.98:
            buckets[2] = (buckets[2][0], buckets[2][1] + 1)
        elif 0.98 <= c <= 1.0:
            buckets[3] = (buckets[3][0], buckets[3][1] + 1)
    return buckets


def _svg_bar_chart(buckets: list[tuple[str, int]]) -> str:
    max_count = max((c for _lbl, c in buckets), default=1)
    width = 720
    height = 160
    pad = 24
    bar_w = (width - pad * 2) / len(buckets)
    chart_h = height - 55

    rects = []
    labels = []
    for i, (lbl, count) in enumerate(buckets):
        x = pad + i * bar_w + 10
        bw = bar_w - 20
        bh = 0 if max_count == 0 else (count / max_count) * chart_h
        y = pad + (chart_h - bh)
        rects.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="8" fill="#2B6CB0"></rect>')
        labels.append(
            f'<text x="{(x + bw/2):.1f}" y="{pad + chart_h + 22:.1f}" text-anchor="middle" font-size="12" fill="#4A5568">{html.escape(lbl)}</text>'
        )
        labels.append(
            f'<text x="{(x + bw/2):.1f}" y="{pad + chart_h + 40:.1f}" text-anchor="middle" font-size="12" fill="#1A202C">{count}</text>'
        )

    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="Confidence distribution">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#F7FAFC" rx="14"></rect>
  {''.join(rects)}
  {''.join(labels)}
</svg>
""".strip()


def generate_report(
    pipeline_output: dict[str, Any],
    customer_name: str = "Your CRM",
    output_path: str = "audit_report.html",
) -> str:
    today = dt.date.today().strftime("%B %d, %Y")

    matches = list(pipeline_output.get("matches", []))
    stats = pipeline_output.get("stats", {})

    total_analyzed = int(stats.get("total_records_a", 0)) + int(stats.get("total_records_b", 0))
    dupes = len(matches)
    hours_saved = (dupes * 3) / 60.0
    missed_cost = dupes * 200

    top = sorted(matches, key=lambda m: float(m.get("confidence", 0.0)), reverse=True)[:20]
    confs = [float(m.get("confidence", 0.0)) for m in matches if m.get("confidence") is not None]
    buckets = _confidence_buckets(confs)
    chart_svg = _svg_bar_chart(buckets)

    def esc(x: Any) -> str:
        return html.escape("" if x is None else str(x))

    rows_html = []
    for m in top:
        ra = m.get("record_a", {})
        rb = m.get("record_b", {})
        conf = float(m.get("confidence", 0.0))
        reason = str(m.get("reason", ""))
        rows_html.append(
            f"""
            <tr>
              <td class="cell">
                <div class="cell__title">{esc(ra.get('id'))}</div>
                <pre class="record">{esc(ra)}</pre>
              </td>
              <td class="cell">
                <div class="cell__title">{esc(rb.get('id'))}</div>
                <pre class="record">{esc(rb)}</pre>
              </td>
              <td class="cell cell--narrow">
                <span class="{_badge_class(conf)}">{conf:.2f}</span>
                <div class="reason">{esc(reason)}</div>
              </td>
            </tr>
            """.strip()
        )

    html_out = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CRM Deduplication Audit — {esc(customer_name)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
      :root {{
        --navy: #0B1F3B;
        --bg: #F2F5F9;
        --card: #FFFFFF;
        --text: #12202F;
        --muted: #5B6B7C;
        --shadow: 0 10px 30px rgba(15, 23, 42, 0.10);
        --border: rgba(15, 23, 42, 0.08);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        color: var(--text);
        background: var(--bg);
      }}
      header {{
        background: linear-gradient(135deg, var(--navy), #123A6B);
        color: #fff;
        padding: 36px 22px;
      }}
      .wrap {{ max-width: 1100px; margin: 0 auto; }}
      .title {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }}
      .subtitle {{ margin-top: 8px; color: rgba(255,255,255,0.85); }}
      main {{ padding: 22px; }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: -26px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 16px 14px;
        box-shadow: var(--shadow);
      }}
      .card .k {{ color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; }}
      .card .v {{ margin-top: 8px; font-size: 22px; font-weight: 700; }}
      .section {{
        margin-top: 18px;
      }}
      .section h2 {{
        font-size: 16px;
        margin: 16px 0 10px;
        color: #1A202C;
        letter-spacing: -0.01em;
      }}
      .table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border-radius: 14px;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        background: #fff;
      }}
      .table th {{
        text-align: left;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #4A5568;
        padding: 12px 14px;
        background: #F7FAFC;
        border-bottom: 1px solid var(--border);
      }}
      .table td {{
        vertical-align: top;
        padding: 14px;
        border-bottom: 1px solid var(--border);
      }}
      .cell__title {{
        font-weight: 700;
        font-size: 12px;
        color: #2D3748;
        margin-bottom: 8px;
      }}
      pre.record {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        color: #1A202C;
        background: #F7FAFC;
        border: 1px solid rgba(15, 23, 42, 0.07);
        border-radius: 12px;
        padding: 10px;
      }}
      .cell--narrow {{ width: 220px; }}
      .badge {{
        display: inline-block;
        font-weight: 700;
        font-size: 12px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(15, 23, 42, 0.10);
      }}
      .badge--green {{ background: rgba(34, 197, 94, 0.12); color: #166534; }}
      .badge--yellow {{ background: rgba(234, 179, 8, 0.16); color: #854d0e; }}
      .badge--gray {{ background: rgba(148, 163, 184, 0.20); color: #334155; }}
      .reason {{
        margin-top: 10px;
        color: #4A5568;
        font-size: 12px;
        line-height: 1.35;
      }}
      footer {{
        color: #6B7280;
        font-size: 12px;
        padding: 26px 22px 36px;
      }}
      @media (max-width: 960px) {{
        .grid {{ grid-template-columns: repeat(2, 1fr); }}
      }}
      @media (max-width: 560px) {{
        .grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <header>
      <div class="wrap">
        <div class="title">CRM Deduplication Audit — {esc(customer_name)}</div>
        <div class="subtitle">{esc(today)}</div>
      </div>
    </header>
    <main>
      <div class="wrap">
        <div class="grid">
          <div class="card">
            <div class="k">Total records analyzed</div>
            <div class="v">{total_analyzed:,}</div>
          </div>
          <div class="card">
            <div class="k">Duplicate pairs found</div>
            <div class="v">{dupes:,}</div>
          </div>
          <div class="card">
            <div class="k">Estimated hours saved</div>
            <div class="v">{hours_saved:,.1f}</div>
          </div>
          <div class="card">
            <div class="k">Estimated cost of missed duplicates</div>
            <div class="v">${missed_cost:,.0f}</div>
          </div>
        </div>

        <div class="section">
          <h2>Top 20 match examples</h2>
          <table class="table">
            <thead>
              <tr>
                <th>Record A</th>
                <th>Record B</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows_html) if rows_html else '<tr><td colspan="3" style="padding:16px;color:#4A5568;">No matches to display.</td></tr>'}
            </tbody>
          </table>
        </div>

        <div class="section">
          <h2>Confidence distribution</h2>
          <div class="card">
            {chart_svg}
          </div>
        </div>
      </div>
    </main>
    <footer>
      <div class="wrap">Generated by EntityMatch API — contact@yourdomain.com</div>
    </footer>
  </body>
</html>
"""

    out_path = Path(output_path)
    out_path.write_text(html_out, encoding="utf-8")
    return str(out_path)

