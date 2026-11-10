"""macOS のカレンダー・リマインダー連携(W4 11/07)。

  - 会話を始める前に「予定から」を押すと、今の(または15分以内に始まる)予定の題と出席者を読み、
    会議の題・参加者の名前の候補にする(終日の予定は使わない)。
  - 会議メモの「やること」をリマインダー(「Sokutsu」リスト)に送る。期限は「11/14」「2026-12-01」
    「1月10日」「金曜」「明日」「Friday」などを日付に直し、読めないものは本文に残す。

どちらも利用者がボタンを押した時だけ動く(起動時や会話の開始で勝手に読まない。初回に macOS の
許可の確認が出るため)。

裏の道:
  1) EventKit(pyobjc-framework-EventKit が入っていれば): 繰り返しの予定も正しく展開される。
  2) osascript の JavaScript(JXA)で Calendar.app / Reminders.app を操作(今の .venv はこちら)。
     Calendar.app の AppleScript は繰り返しの予定の2回目以降を返さない(古くからの制限)。
     毎週の定例を拾いたいなら `pip install pyobjc-framework-EventKit==12.2.2` を勧める。
  アプリとして入れた時(scripts/install_app.sh)は Info.plist に NSAppleEventsUsageDescription・
  NSCalendarsUsageDescription・NSRemindersUsageDescription(とその FullAccess 版)が要る。
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import threading

WINDOW_MIN = 15
LIST_NAME = "Sokutsu"


class ScriptError(RuntimeError):
    pass


def run_osascript(script: str, args: list, timeout: float) -> str:
    r = subprocess.run(["osascript", "-l", "JavaScript", "-e", script, *args],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise ScriptError((r.stderr or r.stdout).strip()[:300])
    return r.stdout.strip()


def _err_kind(e: Exception) -> str:
    s = str(e)
    if "-1743" in s or "Not authorized" in s or "not allowed" in s.lower():
        return "denied"
    if isinstance(e, subprocess.TimeoutExpired):
        return "timeout"
    return "error"


def _when(s) -> "dt.datetime | None":
    try:
        d = dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone().replace(tzinfo=None) if d.tzinfo else d


def pick_event(events: list, now: dt.datetime, window_min: int = WINDOW_MIN) -> "dict | None":
    """今の予定(始まっていて終わっていない)のうち始まりが最も近いもの。無ければ window_min 分以内に
    始まる予定。終日の予定は使わない。出席者は名前かメールのあるものだけ。"""
    cand = []
    for e in events or []:
        if e.get("allday"):
            continue
        a, b = _when(e.get("start")), _when(e.get("end"))
        if a is None or b is None:
            continue
        if a <= now < b:
            cand.append((0, abs((now - a).total_seconds()), e, a, b))
        elif now < a <= now + dt.timedelta(minutes=window_min):
            cand.append((1, (a - now).total_seconds(), e, a, b))
    if not cand:
        return None
    _k, _d, e, a, b = min(cand, key=lambda x: (x[0], x[1]))
    att = []
    for x in e.get("attendees") or []:
        nm, em = str(x.get("name") or "").strip(), str(x.get("email") or "").strip()
        if nm or em:
            att.append({k: v for k, v in (("name", nm), ("email", em)) if v})
    return {"title": str(e.get("title") or "").strip(), "start": a.strftime("%Y-%m-%d %H:%M"),
            "end": b.strftime("%Y-%m-%d %H:%M"), "calendar": e.get("calendar", ""),
            "location": e.get("location", ""), "attendees": att}


# ---------------- カレンダー ----------------

_JXA_EVENTS = r"""
function run(argv) {
  var win = parseInt(argv[0] || "15", 10) * 60 * 1000;
  var Cal = Application("Calendar");
  var now = new Date(), a = new Date(now.getTime() - 12 * 3600 * 1000), b = new Date(now.getTime() + win);
  var out = [];
  Cal.calendars().forEach(function (c) {
    var evs = [];
    try { evs = c.events.whose({_and: [{startDate: {_lessThan: b}}, {endDate: {_greaterThan: now}}]})(); }
    catch (e) { return; }
    evs.forEach(function (ev) {
      try {
        var att = [];
        try { att = ev.attendees().map(function (x) { return {name: x.displayName() || "", email: x.email() || ""}; }); }
        catch (e) {}
        out.push({title: ev.summary() || "", start: ev.startDate().toISOString(), end: ev.endDate().toISOString(),
                  allday: !!ev.alldayEvent(), calendar: c.name(), location: ev.location() || "", attendees: att});
      } catch (e) {}
    });
  });
  return JSON.stringify(out);
}
"""


def _has_eventkit() -> bool:
    try:
        import EventKit  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _ek_store(kind: int):
    """EventKit の許可を取って store を返す(kind 0 = 予定, 1 = リマインダー)。許可が無ければ None。"""
    import EventKit
    store = EventKit.EKEventStore.alloc().init()
    done, ok = threading.Event(), [False]

    def cb(granted, _err):
        ok[0] = bool(granted)
        done.set()
    if kind == 0 and hasattr(store, "requestFullAccessToEventsWithCompletion_"):
        store.requestFullAccessToEventsWithCompletion_(cb)
    elif kind == 1 and hasattr(store, "requestFullAccessToRemindersWithCompletion_"):
        store.requestFullAccessToRemindersWithCompletion_(cb)
    else:
        store.requestAccessToEntityType_completion_(kind, cb)
    done.wait(60)
    return store if ok[0] else None


def _events_eventkit(now: dt.datetime, window_min: int) -> list:
    import EventKit
    from Foundation import NSDate
    store = _ek_store(0)
    if store is None:
        raise ScriptError("Not authorized (-1743)")
    a = NSDate.dateWithTimeIntervalSince1970_((now - dt.timedelta(hours=12)).timestamp())
    b = NSDate.dateWithTimeIntervalSince1970_((now + dt.timedelta(minutes=window_min)).timestamp())
    pred = store.predicateForEventsWithStartDate_endDate_calendars_(a, b, None)
    out = []
    for e in store.eventsMatchingPredicate_(pred) or []:
        att = [{"name": str(x.name() or ""), "email": str(x.URL().resourceSpecifier() or "") if x.URL() else ""}
               for x in (e.attendees() or [])]
        out.append({"title": str(e.title() or ""), "allday": bool(e.isAllDay()),
                    "start": dt.datetime.fromtimestamp(e.startDate().timeIntervalSince1970()).isoformat(),
                    "end": dt.datetime.fromtimestamp(e.endDate().timeIntervalSince1970()).isoformat(),
                    "calendar": str(e.calendar().title() or ""), "location": str(e.location() or ""),
                    "attendees": att})
    return out


def events_now(runner=run_osascript, now: "dt.datetime | None" = None, backend: str = "auto",
               window_min: int = WINDOW_MIN, timeout: float = 25.0) -> dict:
    """{event: 今の予定 or None, backend, error?: "denied"|"timeout"|"error"}。"""
    now = now or dt.datetime.now()
    use = "eventkit" if backend == "eventkit" or (backend == "auto" and _has_eventkit()) else "osascript"
    try:
        if use == "eventkit":
            evs = _events_eventkit(now, window_min)
        else:
            evs = json.loads(runner(_JXA_EVENTS, [str(window_min)], timeout) or "[]")
    except Exception as e:  # noqa: BLE001
        return {"event": None, "backend": use, "error": _err_kind(e), "detail": str(e)[:200]}
    return {"event": pick_event(evs, now, window_min), "backend": use}


# ---------------- リマインダー ----------------

_WD = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6,
       "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
       "周一": 0, "周二": 1, "周三": 2, "周四": 3, "周五": 4, "周六": 5, "周日": 6,
       "星期一": 0, "星期二": 1, "星期三": 2, "星期四": 3, "星期五": 4, "星期六": 5, "星期日": 6,
       "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, "금요일": 4, "토요일": 5, "일요일": 6}


def parse_due(text: str, today: dt.date) -> "dt.date | None":
    """期限の文字 → 日付(読めなければ None)。月日だけで過ぎていれば来年、曜日は次のその曜日。"""
    t = str(text or "").strip()
    if not t:
        return None
    m = re.search(r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})", t)
    try:
        if m:
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        m = re.search(r"(?<!\d)(\d{1,2})\s*(?:/|月)\s*(\d{1,2})(?!\d)", t)
        if m:
            d = dt.date(today.year, int(m[1]), int(m[2]))
            return d if d >= today else dt.date(today.year + 1, int(m[1]), int(m[2]))
    except ValueError:
        return None
    low = t.lower()
    if re.search(r"明日|tomorrow|明天|내일", low):
        return today + dt.timedelta(days=1)
    if re.search(r"今日|today|今天|오늘", low):
        return today
    for k, wd in sorted(_WD.items(), key=lambda kv: -len(kv[0])):
        hit = (re.search(rf"\b{k}", low) if k.isascii() else
               (k in t and (len(k) > 1 or re.search(rf"{k}曜", t))))
        if hit:
            n = (wd - today.weekday()) % 7 or 7
            return today + dt.timedelta(days=n)
    return None


_JXA_REMIND = r"""
function run(argv) {
  var p = JSON.parse(argv[0]);
  var R = Application("Reminders");
  var lst = R.lists.whose({name: p.list})();
  var L = lst.length ? lst[0] : null;
  if (!L) { L = R.List({name: p.list}); R.lists.push(L); L = R.lists.whose({name: p.list})()[0]; }
  var n = 0;
  p.items.forEach(function (it) {
    var props = {name: it.name, body: it.body};
    if (it.due) { var d = new Date(it.due + "T09:00:00"); props.dueDate = d; }
    L.reminders.push(R.Reminder(props));
    n++;
  });
  return String(n);
}
"""


def reminder_items(todos: list, meeting_title: str = "", names: "dict | None" = None,
                   today: "dt.date | None" = None) -> list:
    today = today or dt.date.today()
    names = names or {}
    out = []
    for t in todos:
        task = str(t.get("task") or "").strip()
        if not task:
            continue
        due = parse_due(t.get("due", ""), today)
        body = []
        if t.get("owner"):
            body.append(f"担当: {names.get(t['owner'], t['owner'])}")
        if t.get("due") and not due:
            body.append(f"期限: {t['due']}")
        if meeting_title:
            body.append(f"会議: {meeting_title}")
        if t.get("quote"):
            body.append(f"「{t['quote']}」")
        out.append({"name": task[:200], "body": "\n".join(body), "due": due.isoformat() if due else None})
    return out


def add_reminders(todos: list, meeting_title: str = "", names: "dict | None" = None, runner=run_osascript,
                  today: "dt.date | None" = None, backend: str = "auto", list_name: str = LIST_NAME,
                  timeout: float = 30.0) -> dict:
    """やること → リマインダー。{added, backend, error?}。"""
    items = reminder_items(todos, meeting_title, names, today)
    if not items:
        return {"added": 0, "backend": ""}
    use = "eventkit" if backend == "eventkit" or (backend == "auto" and _has_eventkit()) else "osascript"
    try:
        if use == "eventkit":
            n = _reminders_eventkit(items, list_name)
        else:
            n = int(runner(_JXA_REMIND, [json.dumps({"list": list_name, "items": items}, ensure_ascii=False)],
                           timeout) or 0)
    except Exception as e:  # noqa: BLE001
        return {"added": 0, "backend": use, "error": _err_kind(e), "detail": str(e)[:200]}
    return {"added": n, "backend": use}


def _reminders_eventkit(items: list, list_name: str) -> int:
    import EventKit
    from Foundation import NSDateComponents
    store = _ek_store(1)
    if store is None:
        raise ScriptError("Not authorized (-1743)")
    cal = next((c for c in store.calendarsForEntityType_(1) if str(c.title()) == list_name), None)
    if cal is None:
        cal = EventKit.EKCalendar.calendarForEntityType_eventStore_(1, store)
        cal.setTitle_(list_name)
        src = store.defaultCalendarForNewReminders()
        cal.setSource_(src.source() if src is not None else None)
        store.saveCalendar_commit_error_(cal, True, None)
    n = 0
    for it in items:
        r = EventKit.EKReminder.reminderWithEventStore_(store)
        r.setTitle_(it["name"])
        r.setNotes_(it["body"])
        r.setCalendar_(cal)
        if it["due"]:
            y, m, d = (int(x) for x in it["due"].split("-"))
            c = NSDateComponents.alloc().init()
            c.setYear_(y)
            c.setMonth_(m)
            c.setDay_(d)
            c.setHour_(9)
            r.setDueDateComponents_(c)
        ok = store.saveReminder_commit_error_(r, False, None)
        n += 1 if (ok[0] if isinstance(ok, tuple) else ok) else 0
    store.commit_(None)
    return n
