# 評価の結果と実験の準備

- `fleurs/`: FLEURS(CC BY 4.0)で作った合成会話での評価結果。`ab_*.md` が A/B のまとめ
  (`ab_preview.md` = 予想訳の先出し 11/06、`ab_hedge*.md` = 耳の切り替え案(不採用))、
  `results_*/` が各条件の実行結果。作り方は `scripts/eval_conv.py`。
  評価用の会話音声(約1.6GB)は含めない(`eval_conv.py` で作り直せる)。
- `trunc/`: 日本語の文末欠けの調査の準備(未実施)。`units.npz` は同じ196単位の音声(AutoGain 後)、
  `trunc_exp.py` を リモートサーバー で、`analyze.py` を Mac で動かす。
