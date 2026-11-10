"""用語集・耳の直しの誤りの見張り(11/08 IT 用語)。一般の文(FLEURS の参照文)で
  1) 訳の手がかり(glossary_pairs)がどの語で何回出るか(一般の文に IT の訳を押し付けていないか)
  2) 耳の直し(clean_heard)が文を変えた数(一般の文では 0 が望ましい)
を数えて、多い順に出す。--src-root で読む tsuyaku を切り替えて前後を比べる。

  .venv/bin/python scripts/audit_terms.py --fleurs <fleurs のフォルダ> [--src-root ../wt/src]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_conv as EC  # noqa: E402
import eval_terms as ET  # noqa: E402

LANGS = ("ja", "en", "zh", "ko")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", type=Path, required=True)
    ap.add_argument("--src-root", default=None)
    ap.add_argument("--extra", nargs="*", default=[], help="lang:path(1行1文の一般の文を足す)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    ET.use_src(a.src_root)
    from tsuyaku import terms
    texts = {lang: [r["text"] for r in EC.load_tsv(a.fleurs, lang)] for lang in LANGS}
    for x in a.extra:
        lang, p = x.split(":", 1)
        texts[lang] += [s.strip() for s in Path(p).read_text(encoding="utf-8").splitlines() if s.strip()]
    out = {}
    for src in LANGS:
        hint_sents = 0
        forms = collections.Counter()
        changed = []
        for s in texts[src]:
            got = set()
            for tgt in LANGS:
                if tgt != src:
                    for f, t in terms.glossary_pairs(s, src, tgt):
                        got.add(f)
                        forms[(f, tgt, t)] += 1
            hint_sents += bool(got)
            c = terms.clean_heard(s, src)
            if c != s:
                changed.append((s, c))
        out[src] = {"n": len(texts[src]), "hint_sents": hint_sents, "changed": len(changed),
                    "top": [f"{f}->{tgt}:{t} x{n}" for (f, tgt, t), n in forms.most_common(a.top)],
                    "changed_ex": changed[:20]}
        print(f"{src}: {len(texts[src])}文  手がかりが出た文 {hint_sents}  耳の直しで変わった文 {len(changed)}")
        print("   ", "; ".join(out[src]["top"]))
        for s, c in changed[:10]:
            print("    変:", s[:60], "→", c[:60])
    if a.json:
        a.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
