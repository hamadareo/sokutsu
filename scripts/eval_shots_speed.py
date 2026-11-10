"""断片の例(translate.SHOTTURNS)を足したときの初字・全体の秒を、実際の会話の流れで測る(11/10 mac1)。
LM Studio / llama-server の同じサーバーに、例 0 件と N 件を交互に流す(履歴は会話と同じに伸ばす)。
使い方: PYTHONPATH=src python scripts/eval_shots_speed.py --base http://127.0.0.1:1234 --model google/gemma-3-4b --n 3"""
from __future__ import annotations

import argparse
import json
import statistics as S
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    from tsuyaku import translate as T
    items = json.loads((HERE.parent / "eval/fragments/fragments.json").read_text(encoding="utf-8"))["items"]
    long_ = [i for i in items if i["kind"] in ("long", "short")]
    seq = [long_[k] for k in range(0, len(long_), 3)][: a.turns]
    res: dict = {0: [], a.n: []}
    for rnd in range(a.rounds):
        for shots in (0, a.n, 0, a.n):
            T.SHOTTURNS = shots
            hist = []
            T.warm_prompt(a.base, a.model, ("ja", "zh"))
            for it in seq:
                pair = ("ja", "zh")
                msgs = T.build_messages(pair, "", hist, it["text"], it["src"])
                t0 = time.time()
                first = [0.0]
                out = T.translate_stream(a.base, a.model, msgs, lambda d: first.__setitem__(0, first[0] or time.time()))
                res[shots].append((first[0] - t0 if first[0] else -1, time.time() - t0))
                if out:
                    hist.append({"lang": it["src"], "orig": it["text"], "trans": out})
    for k, v in res.items():
        tt = [x[0] for x in v if x[0] > 0]
        print(f"shots={k}: n={len(v)} ttft med {S.median(tt):.3f} p90 {sorted(tt)[int(len(tt)*.9)]:.3f} total med {S.median(x[1] for x in v):.3f}")


if __name__ == "__main__":
    main()
