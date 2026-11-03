"""Build index.html: live progress of the galaxy SFR sweep (machines B and D on this laptop, others when packs arrive).

Reads the experiment repo (job list, results/meta/*.json, run_*_console.log) and writes a single static page in the
style of the KCB / galaxy work-log site. Only CV numbers are shown until results/final.json exists (the one test evaluation).
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
    "B": ("이 노트북 (CPU 16스레드)", "딥러닝·TabPFN 18개, TabM·RealMLP 튜닝 (1차, 남은 모델은 B1–B3로 재배정)"),
    "B1": ("조원 Mac (B1–B3 한 대)", "B 재실행: TabM 3개 + 튜닝, MLP 2개, RealMLP(log) F5LS"),
    "B2": ("조원 Mac (B1–B3 한 대)", "B 재실행: RealMLP 3개 + 튜닝"),
    "B3": ("조원 Mac (B1–B3 한 대)", "B 재실행: xRFM 3개, TabPFN 3개"),
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


WHY_DEFER = {"TabM ignores the CPU time cap": "TabM은 CPU에서 시간 제한이 적용되지 않음",
             "extreme preset needs a CUDA GPU": "extreme 프리셋은 CUDA GPU 필요"}
WHY_FAIL = [("NaN values in continuous", "결측치를 채우지 않고 넣음 (310af10에서 수정)"),
            ("state_dict to be dict-like", "결측치 때문에 학습값이 깨짐 (310af10에서 수정)"),
            ("not enough memory", "메모리 부족 (RAM 16 GB)")]


def model_failures(failed):
    """One row per unfinished model of the failed local jobs: deferred by the watchdog, or failed with an error."""
    df = LOGS / "deferred.json"
    dd = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
    moved = {n: j[1] for j in J.JOBS if j[1] in ("B1", "B2", "B3") for n in J.names(j)}

    def why_of(n):
        if n in dd:
            return WHY_DEFER.get(dd[n]["reason"], "첫 fold가 60분 넘게 끝나지 않음 (감시 기준)")
        m = re.findall(rf"^{n} FAILED: (\w+)\((.*)\)$", text, re.M)
        return next((k for s_, k in WHY_FAIL if m and s_ in m[-1][1]), m[-1][0] if m else "로그에 오류 없음")

    rows_f = []
    for r in failed:
        job = next(j for j in J.JOBS if j[0] == r["id"])
        lf = LOGS / f"{job[0]}_{job[1]}.log"
        text = lf.read_text(encoding="utf-8", errors="ignore") if lf.exists() else ""
        for n in J.names(job):
            if meta(n):
                continue
            if n in moved:
                rows_f.append((n, r["id"], f'<span class="tag tag-local">{moved[n]} 재배정</span>', why_of(n)))
                continue
            if n in dd:
                v = dd[n]
                why = WHY_DEFER.get(v["reason"], "첫 fold가 60분 넘게 끝나지 않음 (감시 기준)")
                rows_f.append((n, r["id"], '<span class="tag tag-none">GPU로 넘김</span>', why))
                continue
            m = re.findall(rf"^{n} FAILED: (\w+)\((.*)\)$", text, re.M)
            why = next((k for s, k in WHY_FAIL if m and s in m[-1][1]), m[-1][0] if m else "로그에 오류 없음")
            rows_f.append((n, r["id"], '<span class="tag tag-both">오류</span>', why))
    if not rows_f:
        return ""
    body = "".join(f"<tr><td><code>{n}</code></td><td>{j}</td><td>{s}</td><td>{w}</td></tr>" for n, j, s, w in rows_f)
    k_mv = sum("재배정" in s for _, _, s, _ in rows_f)
    k_def = sum("넘김" in s for _, _, s, _ in rows_f)
    return (f"<p>이 노트북에서 끝나지 않은 모델 {len(rows_f)}개: 조원 노트북(B1–B3)으로 재배정 {k_mv}개, "
            f"GPU로 넘김 {k_def}개, 그 밖의 오류 {len(rows_f) - k_mv - k_def}개. 이유 칸은 이 노트북에서 실패하거나 멈춘 원인이고, "
            "결측치 오류는 커밋 310af10에서 고쳤습니다.</p><div class=\"table-wrap\"><table><thead><tr><th>모델</th><th>작업</th><th>상태</th>"
            f"<th>이유</th></tr></thead><tbody>{body}</tbody></table></div>")


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


def ensemble_section(allmeta):
    """Ensemble stage (a04_ensemble.py): method, per-ensemble CV and family weights, selection check, final choice.
    Test scores appear only once results/final.json exists (written after the CV-only choice)."""
    fin_p = EXP / "results" / "final.json"
    intro = ('<p>A·B·C·D의 결과를 모두 모은 뒤, 같은 5-fold의 OOF 예측을 섞는 앙상블 9종을 만들었습니다. 가중치는 CV 안에서 정합니다. '
             'fold k의 예측은 나머지 네 fold로 정한 가중치로 만들기 때문에, 앙상블 CV도 단일 모델과 같은 기준입니다. '
             '기존 6종(simplex 가중치, Caruana greedy, 상위 3·5개 평균, NNLS·LightGBM stacking)에 선행연구에 따른 3종을 더했습니다. '
             'bagged greedy와 sorted initialization(Caruana 등 2006), OOF 상관 0.99 이상의 중복 모델 제거(TabRepo 2024)입니다. '
             '근거는 저장소의 <code>docs/ensemble_literature.md</code>에 있습니다.</p>'
             '<p><strong>최종 선택 규칙 (1-SE, test 평가 전에 커밋)</strong>: CV 1위가 앙상블이고, 그와 fold별 차이가 1 SE 안인 비앙상블 후보가 있으면 '
             '그중 CV가 가장 낮은 모델을 고릅니다(Cawley &amp; Talbot 2010). test 20%는 선택이 끝난 뒤 한 번만 평가합니다.</p>')
    ens = [m for m in allmeta if m.get("stage") == "ensemble"]
    if not ens:
        return intro + '<p class="muted">앙상블 실행 중입니다.</p>'
    rows = []
    for m in ens:
        fam = ", ".join(f"{k} {v:.2f}" for k, v in sorted((m.get("family_weights") or {}).items(), key=lambda x: -x[1]))
        rows.append(f'<tr><td class="mono">{esc(m["name"])}</td><td class="num">{fmt_cv(m)}</td>'
                    f'<td class="num">{len(m.get("weights") or {}) or "—"}</td><td>{esc(fam) or "—"}</td></tr>')
    out = intro + ('<h3>앙상블별 CV</h3><div class="table-wrap"><table><thead><tr><th scope="col">앙상블</th>'
                   '<th scope="col" class="num">CV RMSE</th><th scope="col" class="num">쓴 모델 수</th><th scope="col">계열별 가중치</th>'
                   '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')
    if not fin_p.exists():
        return out + '<p class="muted">최종 선택과 test 평가 진행 중입니다.</p>'
    f = json.loads(fin_p.read_text(encoding="utf-8"))
    sc = f.get("selection_check", {})
    out += ('<h3>최종 모델</h3>'
            f'<p><strong class="mono">{esc(f["final"])}</strong> (CV {f["cv"]:.3f}). CV 1위는 <span class="mono">{esc(f.get("cv_best", f["final"]))}</span>입니다. '
            f'test RMSE <strong>{f["val_rmse"]:.3f}</strong>, MAE {f["val_mae"]:.3f}, R² {f["val_r2"]:.3f} '
            f'(후보 {f["n_candidates"]}개 중 test {f["final_test_rank"]}위; test 순위는 참고용이며 선택에 쓰지 않았습니다).</p>')
    if sc:
        ties = ", ".join(sc.get("within_1se", [])) or "없음"
        out += (f'<p class="small">선택 편향 점검: CV 최저 {sc["cv_min"]:.3f}. 1위를 고르는 절차까지 중첩한 CV {sc["nested_selection_cv"]:.3f}, '
                f'Tibshirani 보정 {sc["tt_corrected_cv"]:.3f}. CV 1위와 1 SE 안의 후보: <span class="mono">{esc(ties)}</span>.</p>')
    trs = "".join(f'<tr><td>{esc(r["role"])}</td><td class="mono">{esc(r["name"])}</td><td class="num">{r["cv"]:.3f}</td>'
                  f'<td class="num">{r["val_rmse"]:.3f}</td><td class="num">{r["val_mae"]:.3f}</td><td class="num">{r["val_r2"]:.3f}</td></tr>'
                  for r in f["table"])
    return out + ('<div class="table-wrap"><table><thead><tr><th scope="col">역할</th><th scope="col">모델</th><th scope="col" class="num">CV RMSE</th>'
                  '<th scope="col" class="num">test RMSE</th><th scope="col" class="num">test MAE</th><th scope="col" class="num">test R²</th>'
                  '</tr></thead><tbody>' + trs + '</tbody></table></div>')


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
          '<p class="small muted">예상 시간은 배정한 기기 기준입니다. B에서 결측치 오류로 실패하거나 CPU에서 너무 오래 걸린 모델은 코드를 고친 뒤(커밋 310af10) 조원 노트북 세 대(B1–B3)에 나눠 다시 돌립니다. B1–B3는 모델마다 시간 상한이 있어, 넘기면 그 모델만 멈추고 다음으로 넘어갑니다.</p>')

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
    mach_of = {m["name"]: ("B·조원 Mac" if r["mach"] in ("B1", "B2", "B3") else r["mach"]) for r in rows for m in r["models"]}
    lrows = []
    for i, m in enumerate(allmeta[:30], 1):
        src = (f"이번 실행 · <strong>{mach_of[m['name']]}</strong>" if m["name"] in mach_of
               else "앙상블" if m.get("stage") == "ensemble" else "이전 실험")
        lrows.append(f"<tr><td class=\"num\">{i}</td><td class=\"mono\">{esc(m['name'])}</td><td>{esc(m.get('features', ''))}</td>"
                     f"<td>{esc(m.get('desc', ''))}</td><td class=\"num\">{fmt_cv(m)}</td><td>{src}</td></tr>")
    s4 = ('<div class="table-wrap"><table><thead><tr><th scope="col" class="num">순위</th><th scope="col">모델</th>'
          '<th scope="col">변수</th><th scope="col">설명</th><th scope="col" class="num">CV RMSE</th><th scope="col">구분</th>'
          '</tr></thead><tbody>' + "".join(lrows) + '</tbody></table></div>'
          f'<p class="small muted">결과가 있는 모델 {len(allmeta)}개 중 CV 상위 30개입니다. 이번 실행 옆의 글자는 그 모델을 돌린 기기(A·B·C·D)입니다. 다른 기기의 결과는 pack 파일을 받아 합치면 여기에 나타납니다.</p>')

    s_ens = ensemble_section(allmeta)

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
    s6 += model_failures(failed)

    secs = [("s-overview", "개요", s1), ("s-machines", "기기별 현황", s2), ("s-jobs", "이 노트북 작업", s3),
            ("s-board", "CV 순위", s4), ("s-ensemble", "앙상블·최종 선택", s_ens), ("s-timeline", "타임라인", s5),
            ("s-rerun", "재실행", s6)]
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
