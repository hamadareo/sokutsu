# Contributing / 貢献のしかた

Sokutsu への参加を歓迎します。小さな直しも、誤字の指摘も、使ってみた感想も助かります。
Contributions of any size are welcome: typo fixes, bug reports, translations, and feedback.

## まずは Issue から / Start with an issue
- **不具合**: [Bug report](../../issues/new?template=bug_report.yml) に、Mac の機種・macOS の版・Sokutsu の版・やったことを書いてください。**会話の内容や名前など、個人の情報は貼らないでください。**
- **要望・質問**: [Feature request](../../issues/new?template=feature_request.yml) か [Discussions](../../discussions) へ。
- 使ってみた感想(どの言語の組み合わせで、どれくらい使えたか)も、とても参考になります。

## コードを送る場合 / Pull requests
1. リポジトリを fork して、ブランチを作ります。
2. テストを流します: `PYTHONPATH=src python -m unittest discover -s tests`
3. 変えた内容と理由を書いて、Pull Request を送ってください。

## 方針 / Ground rules
- 音声と文字を、既定ではネットワークに送らないこと(「このMac」モード)を守ります。
- 実際の会話・本名・サーバーのアドレスなどの個人情報を、コードにもテストにも入れません。
- 使うモデルのライセンスは [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に追記します。
