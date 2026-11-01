"""Build index.html: live progress of the galaxy SFR sweep (machines B and D on this laptop, others when packs arrive).

Reads the experiment repo (job list, results/meta/*.json, run_*_console.log) and writes a single static page in the
style of the KCB / galaxy work-log site. Only CV numbers are shown; test scores stay private until the final run.
usage: python build.py
"""
import html
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SITE = Path(__file__).resolve().parent
EXP = Path(r"C:\Users\ggnat\Documents\Claude\2026-10-31\galaxy-sfr-automl")
META, LOGS = EXP / "results" / "meta", EXP / "results" / "logs"
EVENTS = SITE / "data" / "events.json"
KST = timezone(timedelta(hours=9))
sys.path.insert(0, str(EXP / "scripts"))
import jobs as J  # noqa: E402

MACHINES = {
    "A": ("다른 기기", "단일 모델·선형/커널·변수 선택·AutoGluon 2회"),
    "B": ("이 노트북 (CPU 16스레드)", "딥러닝·TabPFN 18개, TabM·RealMLP 튜닝"),
    "C": ("다른 기기", "트리 모델 nested Optuna 튜닝 + AutoGluon 1회"),
    "D": ("이 노트북 (B 다음 차례)", "AutoGluon 변수 축 F5C·F5LS·F5X·FULL·T2CORE, Tweedie, extreme"),
}
LOCAL = ("B", "D")
WEEK = "월화수목금토일"


def esc(s):
    return html.escape(str(s))


def now():
    return datetime.now(KST)


def meta(name):
    f = META / f"{name}.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text(encoding="utf-8"))
    m["_t"] = datetime.fromtimestamp(f.stat().st_mtime, KST)
    return m


def console(mach):
    f = LOGS / f"run_{mach}_console.log"
    return f.read_text(encoding="utf-8", errors="ignore") if f.exists() else ""


def job_state(job, log):
    """done / failed / running / waiting / queued for one job, from its meta files and its machine's console log."""
    jid, mach = job[0], job[1]
    names = J.names(job)
    if all(meta(n) for n in names):
        return "done"
    if re.search(rf"^{jid} FAILED", log, re.M):
        return "failed"
    if re.search(rf"^=== {jid}:", log, re.M):
        return "running"
    if mach == "D" and "starting D" not in log:
        return "waiting"
    return "queued"


STATE_KO = {"done": "완료", "failed": "실패", "running": "실행 중", "waiting": "대기 (B 종료 후)", "queued": "대기",
            "remote": "결과 미수신"}
STATE_TAG = {"done": "tag-server", "failed": "tag-both", "running": "tag-local", "waiting": "tag-none", "queued": "tag-none",
             "remote": "tag-none"}


def tag(state):
    return f'<span class="tag {STATE_TAG[state]}">{STATE_KO[state]}</span>'


def collect():
    logs = {m: console(m) for m in "ABCD"}
    rows = []
    for job in J.JOBS:
        jid, mach, script, args, res, hrs = job
        if mach == "X":
            continue
        st = job_state(job, logs[mach]) if mach in LOCAL else ("done" if all(meta(n) for n in J.names(job)) else "remote")
        models = []
        first_open = True
        for n in J.names(job):
            m = meta(n)
            if m:
                ms = "done"
            elif st == "running" and first_open:
                ms, first_open = "running", False
            elif st == "failed":
                ms = "failed"
            else:
                ms = st if st in ("waiting", "remote") else "queued"
            models.append({"name": n, "state": ms, "meta": m})
        rows.append({"id": jid, "mach": mach, "script": Path(script).name, "hrs": hrs, "state": st, "models": models})
    return rows, logs


def update_events(rows):
    ev = json.loads(EVENTS.read_text(encoding="utf-8")) if EVENTS.exists() else {}
    t = now().isoformat(timespec="minutes")
    for r in rows:
        if r["mach"] not in LOCAL:
            continue
        if r["state"] in ("running", "done", "failed") and f"start:{r['id']}" not in ev:
            ev[f"start:{r['id']}"] = t
        if r["state"] in ("done", "failed") and f"end:{r['id']}" not in ev:
            ev[f"end:{r['id']}"] = t
        for m in r["models"]:
            if m["meta"] and f"done:{m['name']}" not in ev:
                ev[f"done:{m['name']}"] = m["meta"]["_t"].isoformat(timespec="minutes")
    EVENTS.parent.mkdir(exist_ok=True)
    EVENTS.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    return ev


def fmt_cv(m):
    return f'{m["cv_rmse_mean"]:.3f} <span class="muted">± {m["cv_rmse_sd"]:.2f}</span>'


def sec(n, sid, title, body):
    return (f'<section id="{sid}" aria-labelledby="{sid}-h"><div class="sec-head"><span class="n">{n:02d}</span>'
            f'<h2 id="{sid}-h">{title}</h2></div>\n{body}\n</section>')


def build():
    rows, logs = collect()
    ev = update_events(rows)
    sweep = [m for r in rows for m in r["models"]]
    done = [m for m in sweep if m["meta"]]
    local = [m for r in rows if r["mach"] in LOCAL for m in r["models"]]
    local_done = [m for m in local if m["meta"]]
    best = min(done, key=lambda m: m["meta"]["cv_rmse_mean"]) if done else None
    allmeta = [meta(f.stem) for f in META.glob("*.json")]
    allmeta = sorted([m for m in allmeta if m and "cv_rmse_mean" in m], key=lambda m: m["cv_rmse_mean"])
    lead = allmeta[0] if allmeta else None
    t = now()
    running = [r for r in rows if r["state"] == "running"]
    failed = [r for r in rows if r["state"] == "failed"]
    status = "완료" if all(r["state"] == "done" for r in rows if r["mach"] in LOCAL) else "진행 중"

    # 01 overview
    s1 = f"""<div class="formula" style="margin:0 0 18px;white-space:normal;font-family:inherit;line-height:1.75;overflow-wrap:anywhere"><strong>지금 상황 ({t:%Y-%m-%d %H:%M} KST)</strong> —
이 노트북에서 기기 <strong>B</strong>(딥러닝·TabPFN)를 돌리고 있고, B가 끝나면 기기 <strong>D</strong>(AutoGluon 변수 축)가 자동으로 이어집니다.
{('현재 실행: <strong>' + ', '.join(r['id'] for r in running) + '</strong>. ') if running else ''}이 노트북 몫 {len(local_done)}/{len(local)}개 모델 완료.
{('실패: <strong>' + ', '.join(r['id'] for r in failed) + '</strong> (06 재실행 참고).') if failed else ''}</div>
<dl class="deflist">
  <div><dt>과제</dt><dd>SNU DataHub 천문 데이터(CROCODILE 시뮬레이션 은하 208,437개)로 은하의 <strong>별 생성률(SFR)</strong>을 예측합니다. 점수는 test의 선형 SFR RMSE입니다.</dd></div>
  <div><dt>분할</dt><dd>양식 고정 80/20(train_test_split, seed 42, SFR&gt;0 층화). 20%는 <strong>test</strong>로 마지막에 한 번만 평가하고, 학습 80% 안의 11층 층화 5-fold CV가 validation을 맡습니다.</dd></div>
  <div><dt>선택 규칙</dt><dd>모든 후보(단일·튜닝·AutoGluon·앙상블) 중 <strong>CV RMSE 1위</strong>를 최종 모델로 고릅니다. 후보 전체의 test 점수는 마지막에 표로 공개합니다.</dd></div>
  <div><dt>이 페이지</dt><dd>10분마다 자동 갱신됩니다. CV 숫자만 싣고, test 점수와 예측값·원본 자료는 싣지 않습니다.</dd></div>
</dl>"""

    # 02 machines
    mrows = []
    for mach, (where, what) in MACHINES.items():
        rs = [r for r in rows if r["mach"] == mach]
        ms = [m for r in rs for m in r["models"]]
        nd = sum(1 for m in ms if m["meta"])
        if mach in LOCAL:
            st = ("done" if all(r["state"] == "done" for r in rs) else "running" if any(r["state"] == "running" for r in rs)
                  else "failed" if any(r["state"] == "failed" for r in rs) and all(r["state"] in ("done", "failed") for r in rs)
                  else "waiting" if mach == "D" else "queued")
        else:
            st = "done" if nd == len(ms) else "remote"
        mrows.append(f"<tr><th scope=\"row\">{mach}</th><td>{esc(where)}</td><td>{esc(what)}</td>"
                     f"<td>{' '.join(r['id'] for r in rs)}</td><td class=\"num\">{nd}/{len(ms)}</td>"
                     f"<td class=\"num\">{sum(r['hrs'] for r in rs):.1f} h</td><td>{tag(st)}</td></tr>")
    s2 = ('<div class="table-wrap"><table><thead><tr><th scope="col">기기</th><th scope="col">위치</th><th scope="col">내용</th>'
          '<th scope="col">작업</th><th scope="col" class="num">모델</th><th scope="col" class="num">예상</th>'
          '<th scope="col">상태</th></tr></thead><tbody>' + "".join(mrows) + "</tbody></table></div>"
          '<p class="small muted">예상 시간은 원래 배정한 기기 기준입니다. B는 Mac GPU용으로 잡은 작업을 CPU로 돌리므로 실제로는 더 오래 걸립니다(15–25시간 예상).</p>')

    # 03 jobs (this laptop)
    jparts = []
    for r in rows:
        if r["mach"] not in LOCAL:
            continue
        items = []
        for m in r["models"]:
            mm = m["meta"]
            items.append(f"<tr><td class=\"mono\">{esc(m['name'])}</td><td>{esc(mm.get('features', '')) if mm else ''}</td>"
                         f"<td class=\"num\">{fmt_cv(mm) if mm else '—'}</td>"
                         f"<td class=\"num\">{(str(round(mm.get('seconds', 0) / 60)) + '분') if mm else '—'}</td><td>{tag(m['state'])}</td></tr>")
        nd = sum(1 for m in r["models"] if m["meta"])
        jparts.append(f'<h3>{r["id"]} <span class="muted small">기기 {r["mach"]} · {esc(r["script"])} · {nd}/{len(r["models"])}</span> {tag(r["state"])}</h3>'
                      '<div class="table-wrap"><table><thead><tr><th scope="col">모델</th><th scope="col">변수</th>'
                      '<th scope="col" class="num">CV RMSE</th><th scope="col" class="num">시간</th><th scope="col">상태</th></tr></thead><tbody>'
                      + "".join(items) + "</tbody></table></div>")
    s3 = "\n".join(jparts)

    # 04 leaderboard (all CV results so far)
    sweep_names = {m["name"] for m in sweep}
    lrows = []
    for i, m in enumerate(allmeta[:30], 1):
        src = "이번 실행" if m["name"] in sweep_names else "이전 실험"
        lrows.append(f"<tr><td class=\"num\">{i}</td><td class=\"mono\">{esc(m['name'])}</td><td>{esc(m.get('features', ''))}</td>"
                     f"<td>{esc(m.get('desc', ''))}</td><td class=\"num\">{fmt_cv(m)}</td><td>{src}</td></tr>")
    s4 = ('<div class="table-wrap"><table><thead><tr><th scope="col" class="num">순위</th><th scope="col">모델</th>'
          '<th scope="col">변수</th><th scope="col">설명</th><th scope="col" class="num">CV RMSE</th><th scope="col">구분</th>'
          '</tr></thead><tbody>' + "".join(lrows) + '</tbody></table></div>'
          f'<p class="small muted">결과가 있는 모델 {len(allmeta)}개 중 CV 상위 30개입니다. 다른 기기의 결과는 pack 파일을 받아 합치면 여기에 나타납니다.</p>')

    # 05 timeline
    items = []
    for k, v in ev.items():
        kind, key = k.split(":", 1)
        tt = datetime.fromisoformat(v)
        if kind == "done":
            m = meta(key)
            title, body = f"{key} 완료", (f"CV RMSE {m['cv_rmse_mean']:.3f} (± {m['cv_rmse_sd']:.2f}), {m.get('seconds', 0) / 60:.0f}분. "
                                         f"{esc(m.get('desc', ''))}" if m else "")
        elif kind == "start":
            title, body = f"{key} 시작", ""
        else:
            r = next((r for r in rows if r["id"] == key), None)
            title, body = f"{key} {'실패' if r and r['state'] == 'failed' else '종료'}", ""
        items.append((tt, title, body))
    items.sort(key=lambda x: x[0], reverse=True)
    tl, day = [], None
    for tt, title, body in items:
        d = f"{tt:%Y. %m. %d} ({WEEK[tt.weekday()]})"
        if d != day:
            tl.append(f'<p class="tl-day">{d}</p>')
            day = d
        tl.append(f'<article class="tl-item"><div class="tl-when"><time class="tl-time" datetime="{tt.isoformat()}">{tt:%H:%M}</time></div>'
                  f'<div class="tl-body"><div class="tl-head"><h3 class="tl-title">{esc(title)}</h3><span class="tag tag-local">이 노트북</span></div>'
                  f'<div class="tl-content"><p>{body}</p></div></div></article>')
    s5 = '<div class="timeline">' + "".join(tl) + "</div>" if tl else '<p class="muted">아직 기록이 없습니다.</p>'

    # 06 rerun
    if failed:
        ids = " ".join(r["id"] for r in failed)
        s6 = (f"<p>실패한 작업: <strong>{ids}</strong>. GPU와 RAM 32 GB 이상인 컴퓨터에서 재실행 프롬프트(<code>프롬프트_재실행.md</code>)의 "
              f"첫 줄에 이 번호를 넣어 돌리면 됩니다.</p><pre><code>python scripts/jobs.py run {ids}\npython scripts/jobs.py pack {ids}</code></pre>")
    else:
        s6 = ('<p>아직 실패한 작업이 없습니다. 실패하면 이곳에 번호와 재실행 명령이 나타납니다.</p>'
              '<p class="small muted">미리 알려진 위험: J17(extreme_quality)은 CUDA GPU가 필요해 이 노트북에서 실패할 가능성이 크고, '
              'J15(FULL 564변수)는 RAM 16 GB에서 메모리가 부족할 수 있습니다.</p>')
    df = LOGS / "deferred.json"
    dd = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
    if dd:
        why = {"TabM ignores the CPU time cap": "TabM은 CPU에서 시간 제한이 적용되지 않음",
               "extreme preset needs a CUDA GPU": "extreme 프리셋은 CUDA GPU 필요"}
        rows_d = "".join(f"<tr><td><code>{n}</code></td><td>{why.get(v['reason'], '진행 정체로 중단: ' + v['reason'])}</td>"
                         f"<td>{v['t'][5:16]}</td></tr>" for n, v in dd.items())
        s6 += ('<p>이 노트북에서 오래 걸려 GPU 재실행으로 넘긴 모델입니다. 진행 기록(<code>status.json</code>)이 정해진 시간 넘게 '
               '멈추면 자동으로 중단하고 이 목록에 넣습니다.</p><div class="table-wrap"><table><thead><tr><th>모델</th>'
               f'<th>이유</th><th>시각</th></tr></thead><tbody>{rows_d}</tbody></table></div>')

    secs = [("s-overview", "개요", s1), ("s-machines", "기기별 현황", s2), ("s-jobs", "이 노트북 작업", s3),
            ("s-board", "CV 순위", s4), ("s-timeline", "타임라인", s5), ("s-rerun", "재실행", s6)]
    toc = "".join(f'<li><a href="#{sid}"><span class="n">{i:02d}</span>{t_}</a></li>' for i, (sid, t_, _) in enumerate(secs, 1))
    body = "\n".join(sec(i, sid, t_, b) for i, (sid, t_, b) in enumerate(secs, 1))
    art = (SITE / "assets" / "hero_art.html").read_text(encoding="utf-8")
    best_txt = f"{best['meta']['cv_rmse_mean']:.3f}" if best else "—"
    lead_txt = f"{lead['cv_rmse_mean']:.3f}" if lead else "—"
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="robots" content="noindex, nofollow">
<meta http-equiv="refresh" content="600">
<title>은하 SFR 실험 진행 현황</title>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.20/400.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fontsource/jetbrains-mono@5.0.20/500.css">
<link rel="stylesheet" href="assets/style.css">
</head>
<body data-project="galaxy">
<nav class="topbar" aria-label="주 메뉴"><div class="wrap wide"><a class="brand" href="./"><span class="mark" aria-hidden="true"></span><span class="brand-name">은하 SFR 실험 진행</span></a><div class="nav-links"><a href="./" aria-current="page">진행 현황</a><a href="https://github.com/Yu-Jusang/galaxy-sfr-automl">코드</a></div></div></nav>
<header class="hero">{art}<div class="wrap wide"><div class="hero-text"><nav class="crumbs" aria-label="경로"><span>은하 SFR</span><span aria-hidden="true"> / </span><span aria-current="page">실행 진행</span></nav><div class="meta-row"><p class="eyebrow">SNU DataHub 경진대회 · 팀 AutoML 실행</p><span class="status">{status}</span></div><h1>은하 SFR 실험 진행 현황</h1><p class="summary">네 기기에 나눈 모델·튜닝·AutoGluon 실험의 진행 상황입니다. 이 노트북은 기기 B와 D를 맡아 차례로 돌립니다.</p><dl class="metrics"><div><dt>이 노트북 완료</dt><dd>{len(local_done)}/{len(local)}</dd></div><div><dt>이번 실행 CV 1위</dt><dd>{best_txt}</dd></div><div><dt>전체 CV 1위</dt><dd>{lead_txt}</dd></div></dl><p class="note">{esc(best['name']) + ' · ' if best else ''}전체 1위 {esc(lead['name']) if lead else '—'} · 마지막 갱신 {t:%Y-%m-%d %H:%M} KST</p></div></div></header>
<div class="wrap wide layout"><nav class="toc" aria-label="목차"><ol>{toc}</ol></nav><main class="content">
{body}
</main></div>
<footer class="wrap wide"><p class="small muted">자동 생성 · build.py · {t:%Y-%m-%d %H:%M} KST</p></footer>
</body>
</html>
"""
    (SITE / "index.html").write_text(page, encoding="utf-8")
    print("built", SITE / "index.html", f"{len(local_done)}/{len(local)}")


if __name__ == "__main__":
    build()
