"""W4 macOS のカレンダー・リマインダー連携(11/07)。本物の Calendar/Reminders は呼ばない(許可の
確認が画面に出るため)。osascript を偽物に差し替えて、送る台本と結果の読み方・選び方を確かめる。"""
import datetime as dt
import json
import unittest

from tsuyaku import calendar_mac as C

NOW = dt.datetime(2026, 11, 7, 10, 5)


def ev(title, start, end, allday=False, att=()):
    return {"title": title, "start": start.isoformat(), "end": end.isoformat(), "allday": allday,
            "calendar": "仕事", "location": "", "attendees": [{"name": n, "email": e} for n, e in att]}


class TestPick(unittest.TestCase):
    def test_ongoing_event_wins_over_next_and_all_day_is_skipped(self):
        evs = [ev("終日の予定", NOW.replace(hour=0, minute=0), NOW.replace(hour=23, minute=59), allday=True),
               ev("次の会議", NOW + dt.timedelta(minutes=10), NOW + dt.timedelta(minutes=70)),
               ev("定例", NOW - dt.timedelta(minutes=5), NOW + dt.timedelta(minutes=25),
                  att=[("山田 花子", "h@example.com"), ("", "smith@example.com")])]
        p = C.pick_event(evs, NOW)
        self.assertEqual(p["title"], "定例")
        self.assertEqual([a.get("name") or a.get("email") for a in p["attendees"]],
                         ["山田 花子", "smith@example.com"])

    def test_starting_soon_is_used_when_nothing_is_ongoing(self):
        evs = [ev("あとの会議", NOW + dt.timedelta(minutes=40), NOW + dt.timedelta(minutes=90)),
               ev("もうすぐ", NOW + dt.timedelta(minutes=8), NOW + dt.timedelta(minutes=38))]
        self.assertEqual(C.pick_event(evs, NOW)["title"], "もうすぐ")
        self.assertIsNone(C.pick_event([evs[0]], NOW))         # 15分より先は出さない
        self.assertIsNone(C.pick_event([], NOW))


class TestOsascript(unittest.TestCase):
    def test_calendar_query_uses_jxa_and_parses(self):
        seen = {}

        def fake(script, args, timeout):
            seen["script"], seen["args"] = script, args
            return json.dumps([ev("定例", NOW - dt.timedelta(minutes=1), NOW + dt.timedelta(minutes=29))])
        got = C.events_now(runner=fake, now=NOW, backend="osascript")
        self.assertEqual(got["event"]["title"], "定例")
        self.assertIn('Application("Calendar")', seen["script"])
        self.assertEqual(seen["args"], ["15"])
        self.assertEqual(got["backend"], "osascript")

    def test_permission_denied_is_reported_not_raised(self):
        def fake(script, args, timeout):
            raise C.ScriptError("execution error: Not authorized to send Apple events to Calendar. (-1743)")
        got = C.events_now(runner=fake, now=NOW, backend="osascript")
        self.assertIsNone(got["event"])
        self.assertEqual(got["error"], "denied")

    def test_reminders_payload(self):
        sent = {}

        def fake(script, args, timeout):
            sent["script"], sent["payload"] = script, json.loads(args[0])
            return "2"
        todos = [{"task": "見積もりを送る", "owner": "S2", "due": "11/14", "quote": "send it"},
                 {"task": "議事録を共有", "owner": "", "due": "来週のどこか"}]
        n = C.add_reminders(todos, meeting_title="定例", names={"S2": "Smith"}, runner=fake, today=NOW.date(),
                            backend="osascript")
        self.assertEqual(n["added"], 2)
        p = sent["payload"]
        self.assertEqual(p["list"], "Sokutsu")
        self.assertEqual(p["items"][0]["name"], "見積もりを送る")
        self.assertEqual(p["items"][0]["due"], "2026-11-14")
        self.assertIn("Smith", p["items"][0]["body"])
        self.assertIn("定例", p["items"][0]["body"])
        self.assertIsNone(p["items"][1]["due"])                  # 読めない期限は本文に残す
        self.assertIn("来週のどこか", p["items"][1]["body"])
        self.assertIn('Application("Reminders")', sent["script"])


class TestDue(unittest.TestCase):
    def test_parse_due(self):
        d = NOW.date()   # 2026-11-07 は土曜
        self.assertEqual(C.parse_due("2026-12-01", d), dt.date(2026, 12, 1))
        self.assertEqual(C.parse_due("11/14", d), dt.date(2026, 11, 14))
        self.assertEqual(C.parse_due("1月10日まで", d), dt.date(2027, 1, 10))   # 過ぎた月日は来年
        self.assertEqual(C.parse_due("金曜", d), dt.date(2026, 11, 13))
        self.assertEqual(C.parse_due("by Friday", d), dt.date(2026, 11, 13))
        self.assertEqual(C.parse_due("明日", d), dt.date(2026, 11, 8))
        self.assertEqual(C.parse_due("tomorrow", d), dt.date(2026, 11, 8))
        self.assertIsNone(C.parse_due("そのうち", d))
        self.assertIsNone(C.parse_due("", d))


if __name__ == "__main__":
    unittest.main()
