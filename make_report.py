"""Build report.html (final report page) from the experiment repo's docs/FINAL_REPORT.md, in the site's design.
usage: python make_report.py   (needs the markdown package; base Anaconda python has it)
"""
import html
import json
import re
from pathlib import Path

import markdown

SITE = Path(__file__).resolve().parent
EXP = Path(r"C:\Users\ggnat\Documents\Claude\2026-10-31\galaxy-sfr-automl")
md = (EXP / "docs" / "FINAL_REPORT.md").read_text(encoding="utf-8")
fin = json.loads((EXP / "results" / "final.json").read_text(encoding="utf-8"))   # aggregate metrics only

title, rest = md.split("\n", 1)
title = title.lstrip("# ").strip()
parts = re.split(r"^## ", rest, flags=re.M)
lead, secs = parts[0].strip(), parts[1:]


def conv(text):
    text = re.sub(r"^  (?=[-\d])", "    ", text, flags=re.M)   # Python-Markdown nests lists at 4 spaces, the file uses 2
    h = markdown.markdown(text, extensions=["tables", "fenced_code"])
    h = h.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return h.replace("<h3>", '<h3 class="sub">')


blocks, toc = [], []
for i, s in enumerate(secs, 1):
    head, body = s.split("\n", 1)
    sid = f"r{i}"
    toc.append(f'<li><a href="#{sid}"><span class="n">{i:02d}</span>{html.escape(head.strip())}</a></li>')
    blocks.append(f'<section id="{sid}" aria-labelledby="{sid}-h"><div class="sec-head"><span class="n">{i:02d}</span>'
                  f'<h2 id="{sid}-h">{html.escape(head.strip())}</h2></div>\n{conv(body)}\n</section>')

art = (SITE / "assets" / "hero_art.html").read_text(encoding="utf-8")
page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex, nofollow">
<title>은하 SFR 최종 보고서</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.20/400.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.20/500.css">
<link rel="stylesheet" href="assets/style.css">
</head>
<body data-project="galaxy">
<nav class="topbar" aria-label="주 메뉴"><div class="wrap wide"><a class="brand" href="./"><span class="mark" aria-hidden="true"></span><span class="brand-name">은하 SFR 실험 진행</span></a><div class="nav-links"><a href="./">진행 현황</a><a href="report.html" aria-current="page">최종 보고서</a><a href="https://github.com/Yu-Jusang/galaxy-sfr-automl">코드</a></div></div></nav>
<header class="hero">{art}<div class="wrap wide"><div class="hero-text"><nav class="crumbs" aria-label="경로"><span>은하 SFR</span><span aria-hidden="true"> / </span><span aria-current="page">최종 보고서</span></nav><div class="meta-row"><p class="eyebrow">SNU DataHub 경진대회 · 팀 AutoML 실행</p><span class="status">완료</span></div><h1>{html.escape(title)}</h1>
<div class="summary">{conv(lead)}</div>
<dl class="metrics"><div><dt>최종 모델 CV</dt><dd>{fin['cv']:.3f}</dd></div><div><dt>최종 test RMSE</dt><dd>{fin['val_rmse']:.3f}</dd></div><div><dt>R0 test RMSE</dt><dd>{fin['r0_val_rmse']:.3f}</dd></div></dl>
<p class="note">{html.escape(fin['final'])} · 후보 {fin['n_candidates']}개 · test는 선택 뒤 1회 평가</p></div></div></header>
<div class="wrap wide layout"><nav class="toc" aria-label="목차"><ol>{''.join(toc)}</ol></nav><main class="content">
{chr(10).join(blocks)}
</main></div>
<footer class="wrap wide"><p class="small muted">docs/FINAL_REPORT.md에서 생성 · make_report.py</p></footer>
</body>
</html>
"""
(SITE / "report.html").write_text(page, encoding="utf-8")
print("built", SITE / "report.html")
