"""The Practice & mastery mode (modules/practice.py): a short daily review of verified problems with fresh numbers each time, scheduled by
spaced repetition, and a view of what you have mastered. No language model.

The date used everywhere comes from `_today()`: the real date, unless a session sets `_practice_today` (the tests do, and it lets you
rehearse tomorrow's queue)."""
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from modules import history, practice
from modules import problem_sets as ps
from modules.palette import SEQUENTIAL
from ui.a11y import show_figure
from ui.theme import badge_row

_LAST = "_prac_last"            # the feedback on the problem just answered
_EXTRA = "_prac_extra"          # (item id, nonce): an extra, unscheduled variant being practised
_ICON = {"correct": "✅", "close": "🟡", "wrong": "❌", "missing": "⚪", "unreadable": "❓"}
_LEVEL = {"new": ("new", "neutral"), "learning": ("learning", "warn"), "solid": ("solid", "pass"), "mastered": ("mastered", "pass")}


def _today() -> date:
    return st.session_state.get("_practice_today") or date.today()


# ------------------------------------------------------------------------------------------ today

def _feedback(last: dict) -> None:
    item, inst, grade, state = last["item"], last["instance"], last["grade"], last["state"]
    (st.success if grade.all_correct else st.warning)(f"{item.title}: {grade.summary}")
    for target, v in grade.verdicts.items():
        line = f"{_ICON[v.status]} **{target}**: your answer {last['answers'].get(target, '') or '(none)'} · the key is {ps.fmt(inst.answers[target])}"
        if v.diagnosis:
            line += f" · {v.diagnosis}"
        st.markdown(line)
    if last["scheduled"]:
        days = (state.due - _today()).days
        st.caption(f"Next review in {days} day{'s' if days != 1 else ''} ({state.due.strftime('%d %b')}).")
    else:
        st.caption("This was an extra attempt: it is recorded, but the review schedule did not change.")
    c1, c2 = st.columns(2)
    if c1.button("Next problem →", type="primary", key="prac_next"):
        st.session_state.pop(_LAST, None)
        st.rerun()
    if c2.button("Try another variant of this one (extra)", key="prac_extra"):
        nonce = st.session_state.get(_EXTRA, (item.id, 0))[1] + 1 if st.session_state.get(_EXTRA, (None, 0))[0] == item.id else 1
        st.session_state[_EXTRA] = (item.id, nonce)
        st.session_state.pop(_LAST, None)
        st.rerun()


def _question(item: practice.Item, nonce: int, scheduled: bool, kind: str) -> None:
    today = _today()
    inst, seed = practice.variant(item, today, nonce)
    st.markdown(f"##### {item.title}  ·  _{kind}_")
    st.write(inst.text)
    if not inst.rewritten:
        st.caption("Use: " + ", ".join(f"{g['symbol']} = {ps.fmt(g['value'])}{(' ' + g['unit']) if g['unit'] else ''}" for g in inst.given))
    answers: dict[str, str] = {}
    cols = st.columns(len(inst.find))
    for col, f in zip(cols, inst.find):
        label = f"{f['meaning']} ({f['symbol']})" + (f" in {f['unit']}" if f["unit"] else "")
        answers[f["symbol"]] = col.text_input(label, key=f"prac_ans_{item.id}_{item.state.reps}_{nonce}_{f['symbol']}")
    c1, c2 = st.columns([1, 1])
    if c1.button("Check", type="primary", key=f"prac_check_{item.id}_{nonce}"):
        base = ps.generate(item.text, item.raw, n=1, seed=0).base_answers
        grade = practice.grade(inst, answers, base)
        state = practice.record(item.id, seed, grade, today, scheduled)
        st.session_state[_LAST] = dict(item=item, instance=inst, grade=grade, state=state, answers=answers, scheduled=scheduled)
        st.session_state.pop(_EXTRA, None)
        st.rerun()
    if scheduled and c2.button("Skip for now", key=f"prac_skip_{item.id}"):
        st.session_state.setdefault("_prac_skipped", set()).add(item.id)
        st.rerun()


def _today_tab() -> None:
    today = _today()
    s = practice.summary(today)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Streak", f"{s['streak']} day{'s' if s['streak'] != 1 else ''}", help=f"Longest: {s['longest_streak']}")
    c2.metric("Reviews due", s["due"])
    c3.metric("New today", s["new"])
    c4.metric("Accuracy, last 7 days", "—" if s["accuracy_7d"] is None else f"{s['accuracy_7d']:.0%}", help=f"{s['attempts_7d']} attempts")
    if s["items"] == 0:
        st.info("You have no problems to practise yet. Add the starter problems (physics, chemistry, finance), or add problems you have solved from the "
                "'My problems' tab.")
        if st.button("➕ Add the starter problems", type="primary", key="prac_seed"):
            added = practice.seed_library(today)
            st.success(f"Added {added} problems.")
            st.rerun()
        return
    last = st.session_state.get(_LAST)
    if last is not None:
        _feedback(last)
        return
    extra = st.session_state.get(_EXTRA)
    if extra is not None:
        item = practice.get_item(extra[0])
        if item is not None and item.active:
            _question(item, extra[1], False, "extra practice")
            return
        st.session_state.pop(_EXTRA, None)
    skipped = st.session_state.get("_prac_skipped", set())
    entries = [e for e in practice.queue(today) if e.item.id not in skipped]
    if not entries:
        done = practice.summary(today)["practised_today"]
        st.success("You are done for today. " + ("Come back tomorrow." if done else "Nothing is due."))
        upcoming = [(d, n) for d, n in practice.forecast(today, 7) if n]
        if upcoming:
            st.caption("Coming up: " + ", ".join(f"{d.strftime('%a %d %b')}: {n}" for d, n in upcoming))
        if skipped and st.button("Bring back the ones I skipped", key="prac_unskip"):
            st.session_state["_prac_skipped"] = set()
            st.rerun()
        return
    entry = entries[0]
    kind = "new" if entry.kind == "new" else ("review, due today" if entry.overdue_days == 0 else f"review, {entry.overdue_days} day{'s' if entry.overdue_days != 1 else ''} overdue")
    st.caption(f"{len(entries)} left in today's queue.")
    _question(entry.item, 0, True, kind)


# ------------------------------------------------------------------------------------------ mastery

def _mastery_tab() -> None:
    today = _today()
    ms = practice.mastery(today)
    if not ms or all(m.attempts == 0 for m in ms):
        st.info("Mastery appears after your first few attempts.")
        return
    st.markdown("**By concept**, weakest first. Accuracy is weighted by recency (an attempt counts half as much every three weeks), and a concept needs "
                "several attempts before it can be called mastered.")
    for m in ms:
        label, kind = _LEVEL[m.level]
        acc = "—" if m.accuracy is None else f"{m.accuracy:.0%}"
        badge_row([(f"{m.concept}: {label}", kind), (f"accuracy {acc}", "neutral"), (f"{m.attempts} attempt{'s' if m.attempts != 1 else ''}", "neutral")])
    concepts, weeks, matrix = practice.heatmap(today, 8)
    fig = go.Figure(go.Heatmap(z=[[(v if v is not None else None) for v in row] for row in matrix], x=weeks, y=concepts, zmin=0, zmax=1, colorscale=SEQUENTIAL,
                               hoverongaps=False, colorbar=dict(title="average score"),
                               hovertemplate="%{y}<br>week of %{x}<br>%{z:.0%}<extra></extra>"))
    fig.update_layout(height=max(260, 46 * len(concepts) + 120), title="Average score per concept per week (blank: not practised)",
                      xaxis_title="week starting", yaxis=dict(autorange="reversed"))
    show_figure(fig, width="stretch")
    days = practice.forecast(today, 14)
    bar = go.Figure(go.Bar(x=[d.strftime("%a %d") for d, _ in days], y=[n for _, n in days]))
    bar.update_layout(height=260, title="Reviews coming due in the next two weeks", yaxis_title="reviews")
    show_figure(bar, width="stretch")


# ------------------------------------------------------------------------------------------ my problems

def _items_tab() -> None:
    today = _today()
    items = practice.list_items()
    if items:
        st.dataframe([{"problem": i.title, "concepts": ", ".join(i.concepts), "from": i.origin, "reviews": i.state.reps, "lapses": i.state.lapses,
                       "ease": round(i.state.ease, 2), "next review": "not started" if i.is_new else i.state.due.strftime("%d %b %Y"),
                       "last": "—" if i.last_reviewed is None else i.last_reviewed.strftime("%d %b")} for i in items], hide_index=True, width="stretch")
        c1, c2 = st.columns([3, 1])
        gone = c1.selectbox("Stop practising", items, format_func=lambda i: i.title, key="prac_remove_pick", index=None, placeholder="Choose a problem")
        if c2.button("Remove", key="prac_remove", disabled=gone is None) and gone is not None:
            practice.remove_item(gone.id)
            st.rerun()
    st.markdown("#### Add problems")
    a, b = st.columns(2)
    if a.button("➕ Add the starter problems", key="prac_seed2"):
        st.success(f"Added {practice.seed_library(today)} problems.")
        st.rerun()
    if st.session_state.get("model") is not None and st.session_state.get("problem_text"):
        if b.button("➕ Add the problem that is open", key="prac_add_open"):
            try:
                model = st.session_state["model"]
                _, created = practice.add_item(st.session_state["problem_text"][:60], st.session_state["problem_text"], model.raw_json,
                                               [model.problem_domain], "open", today)
                st.success("Added." if created else "It is already in your practice.")
            except practice.PracticeError as exc:
                st.warning(str(exc))
    entries = [e for e in history.list_recent(60) if e["passed"]]
    if entries:
        picks = st.multiselect("From my history (verified problems)", entries, format_func=lambda e: f"#{e['id']} · {e['problem_text'][:70]}", key="prac_hist_pick")
        if picks and st.button("Add the selected", key="prac_hist_add"):
            added, problems = 0, []
            for e in picks:
                loaded = history.load(e["id"])
                if loaded is None:
                    continue
                try:
                    _, created = practice.add_item(loaded[0][:60], loaded[0], loaded[1].raw_json, [loaded[1].problem_domain], "history", today)
                    added += int(created)
                except practice.PracticeError as exc:
                    problems.append(f"#{e['id']}: {exc}")
            if added:
                st.success(f"Added {added}.")
            for msg in problems:
                st.warning(msg)


def render_practice_tab() -> None:
    st.subheader("🧠 Practice & mastery")
    st.caption("A few minutes a day. Each problem comes back on a spaced schedule, with new numbers every time, so you practise the method and not an "
               "answer you remember. Every answer key passed the app's own verification. No language model involved, and nothing leaves this computer.")
    tabs = st.tabs(["Today", "Mastery", "My problems"])
    with tabs[0]:
        _today_tab()
    with tabs[1]:
        _mastery_tab()
    with tabs[2]:
        _items_tab()
