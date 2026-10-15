"""Render the leaderboard as a static site for GitHub Pages.

Deliberately dependency-free: the output is a single self-contained HTML
file with no external CSS or JS, so Pages can serve it from a static host
with no build step and it cannot rot when a CDN changes.

The page states its own caveats. A leaderboard that hides its weaknesses
is worse than none, because reviewers spend time on numbers that will be
quietly revised.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

SITE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dynamion leaderboard</title>
<style>
  :root {{
    --bg: #0d1117; --fg: #c9d1d9; --muted: #8b949e;
    --border: #30363d; --accent: #58a6ff; --warn: #d29922; --ok: #3fb950;
  }}
  body {{
    background: var(--bg); color: var(--fg); margin: 0;
    font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 60rem; margin: 0 auto; padding: 2.5rem 1.5rem; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.02em; }}
  h2 {{ font-size: 1.05rem; margin: 2.5rem 0 .75rem; color: var(--accent); }}
  .sub {{ color: var(--muted); margin: 0 0 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0 0 1rem; }}
  th, td {{ text-align: left; padding: .6rem .75rem; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; font-size: .8rem; text-transform: uppercase;
        letter-spacing: .04em; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  .pill {{ display: inline-block; padding: .1rem .5rem; border-radius: 999px;
           font-size: .72rem; border: 1px solid var(--border); color: var(--muted); }}
  .pill.ok {{ color: var(--ok); border-color: #238636; }}
  .pill.warn {{ color: var(--warn); border-color: #9e6a03; }}
  .bar {{ height: 4px; background: var(--border); border-radius: 2px; overflow: hidden;
          min-width: 80px; }}
  .bar > span {{ display: block; height: 100%; background: var(--accent); }}
  .caveat {{ background: #161b22; border: 1px solid var(--border); border-left: 3px solid var(--warn);
             border-radius: 6px; padding: 1rem 1.25rem; margin: 1rem 0; }}
  .caveat ul {{ margin: .5rem 0 0; padding-left: 1.2rem; color: var(--muted); }}
  .caveat li {{ margin: .3rem 0; }}
  footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border);
            color: var(--muted); font-size: .85rem; }}
  code {{ background: #161b22; padding: .1rem .35rem; border-radius: 4px; font-size: .9em; }}
  .empty {{ color: var(--muted); font-style: italic; }}
</style>
</head>
<body><div class="wrap">
<h1>dynamion</h1>
<p class="sub">Long-horizon software-engineering evaluation for coding agents.</p>
{body}
<footer>
Generated from <code>results/leaderboard.json</code> by
<code>python -m dynamion.site</code>. Reproduce with
<code>dynamion bench --tasks-dir tasks</code>.
</footer>
</div></body>
</html>
"""


def render_table(rows: list[dict]) -> str:
    if not rows:
        return (
            '<p class="empty">No results yet. Run <code>dynamion bench</code> '
            "to populate the leaderboard.</p>"
        )

    head = (
        "<tr><th>Agent</th><th>Resolved</th><th>Score</th>"
        "<th>Tokens</th><th>Wall (s)</th><th>Adapter</th></tr>"
    )
    body = []
    for row in rows:
        verified = row.get("adapter_verified", False)
        pill = (
            '<span class="pill ok">verified</span>'
            if verified
            else '<span class="pill warn">unverified</span>'
        )
        mean = row.get("mean_score", 0.0)
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('agent', '?')))}</td>"
            f'<td class="num">{row.get("resolved", 0)}/{row.get("n", 0)} '
            f'({row.get("resolve_rate", 0.0):.0%})</td>'
            f'<td class="num">{mean:.3f}'
            f'<div class="bar"><span style="width:{mean * 100:.0f}%"></span></div></td>'
            f'<td class="num">{row.get("total_tokens", 0)}</td>'
            f'<td class="num">{row.get("total_wall_seconds", 0):.0f}</td>'
            f"<td>{pill}</td>"
            "</tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def render_failure_modes(rows: list[dict]) -> str:
    seen: dict[str, int] = {}
    for row in rows:
        for mode, count in (row.get("failure_modes") or {}).items():
            seen[mode] = seen.get(mode, 0) + count
    if not seen:
        return '<p class="empty">No failure data yet.</p>'
    items = "".join(
        f"<li><code>{html.escape(mode)}</code> &mdash; {count}</li>"
        for mode, count in sorted(seen.items(), key=lambda kv: -kv[1])
    )
    return f"<ul>{items}</ul>"


def build_html(rows: list[dict]) -> str:
    body = [
        "<h2>Leaderboard</h2>",
        render_table(rows),
        "<h2>Failure modes</h2>",
        render_failure_modes(rows),
        '<div class="caveat"><strong>Read these before quoting a number.</strong>'
        "<ul>"
        "<li>Every score here is from a <em>single</em> agent tier on a small "
        "task set. It establishes that the pipeline works, not anything about "
        "model capability.</li>"
        "<li>Rows marked <em>unverified</em> were not executed against the real "
        "CLI. The adapter ships so the harness is extensible; the number is not "
        "a measurement.</li>"
        "<li>Grading is hermetic, but some past failures were the grader's fault "
        "rather than the agent's. Each such bug is documented in the repository "
        "README with a regression test pinning it.</li>"
        "<li>Rubric subscores are excluded because the available judge model is "
        "too small to rank agents with.</li>"
        "</ul></div>",
    ]
    return SITE_TEMPLATE.format(body="\n".join(body))


def write_site(rows: list[dict], out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(build_html(rows), encoding="utf-8")
    (out_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="dynamion-site")
    parser.add_argument("--results", default="results/leaderboard.json")
    parser.add_argument("--out", default="site")
    args = parser.parse_args(argv)

    path = Path(args.results)
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    index = write_site(rows, Path(args.out))
    print(f"wrote {index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())