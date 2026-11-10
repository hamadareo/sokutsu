# Third-party notices / 第三者のライセンス

Sokutsu のソースコードは [MIT License](LICENSE) です。このファイルは、Sokutsu が利用・同梱・ダウンロードする
第三者のソフトウェアとモデルの一覧(要約)です。**要約であり、法的な助言ではありません。** 各ライセンスの全文と
著作権表示は、配布用アプリ(DMG)の `THIRD_PARTY_NOTICES.txt`(ひな形: [packaging/notices_header.txt](packaging/notices_header.txt))
と、各配布元にあります。確認しきれていないものは **「要確認」** と書いてあります。

This is a summary, not legal advice. Items marked **要確認 (to be verified)** have not been fully checked.

## A. ソフトウェア(配布用アプリに同梱、または pip で入るもの)

| 名前 | 用途 | ライセンス | 配布元 |
|---|---|---|---|
| CPython 3.12(python-build-standalone) | 配布用アプリの Python | PSF-2.0(内包: OpenSSL Apache-2.0、SQLite パブリックドメイン、libffi MIT ほか。詳細は packaging/notices_header.txt) | python.org / astral-sh/python-build-standalone |
| llama.cpp (llama-server) | 配布用アプリの翻訳エンジン | MIT(内包: BoringSSL、cpp-httplib、nlohmann/json、miniaudio、stb_image など) | github.com/ggml-org/llama.cpp |
| sherpa-onnx / sherpa-onnx-core 2.0.1 | SenseVoice の実行・Kokoro の読み上げ | Apache-2.0(内包: ONNX Runtime MIT、OpenFst、kaldi-decoder、kaldi-native-fbank ほか) | github.com/k2-fsa/sherpa-onnx |
| pywhispercpp 1.5.1(whisper.cpp / ggml を内包) | Whisper の実行 | MIT | github.com/absadiki/pywhispercpp |
| sounddevice 0.5.6(PortAudio を内包) | マイク入力 | MIT / PortAudio ライセンス | github.com/spatialaudio/python-sounddevice |
| NumPy 2.2.6 | 数値計算 | BSD-3-Clause(内包: OpenBLAS、LAPACK、GCC ランタイム GPL-3.0 with GCC exception、libquadmath LGPL-2.1+) | numpy.org |
| pywebview 6.2.1 | 画面 | BSD-3-Clause | github.com/r0x0r/pywebview |
| bottle 0.13.4 | pywebview の依存 | MIT | github.com/bottlepy/bottle |
| proxy_tools 0.1.0 | pywebview の依存 | MIT(sdist にライセンスファイルなし。PyPI のメタデータによる。**要確認**) | github.com/jtushman/proxy_tools |
| requests 2.34.2 | HTTP | Apache-2.0 | github.com/psf/requests |
| certifi / tqdm | requests・進捗表示 | MPL-2.0(tqdm は MPL-2.0 AND MIT) | github.com/certifi/python-certifi / github.com/tqdm/tqdm |
| urllib3、idna、charset-normalizer、cffi、pycparser、typing_extensions、platformdirs | requests などの依存 | MIT / BSD-3-Clause / MIT-0 / PSF-2.0(版は requirements-dist.txt) | 各 PyPI ページ |
| PyObjC(pyobjc-core と pyobjc-framework-*) | macOS API(EventKit など) | MIT | github.com/ronaldoussoren/pyobjc |
| faster-whisper(任意) | `server/remote_stt_server.py`(リモートの耳)。リポジトリには含めず、利用者が自分のサーバーに入れる | MIT(CTranslate2 も MIT。**要確認**: 使う版で確認) | github.com/SYSTRAN/faster-whisper |

## B. モデル(リポジトリには含まない。公式の配布元から取得)

| モデル | 用途 | ライセンス・条件 | 取得元 |
|---|---|---|---|
| Gemma 3 4B IT QAT (Q4_0 GGUF) | 既定の翻訳モデル(配布用アプリ) | **Gemma Terms of Use** と Gemma Prohibited Use Policy(https://ai.google.dev/gemma/terms)。再配布するときは条件の引き継ぎ・利用規約の写し・変更の明示・「Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms」の表示が必要 | huggingface.co/ggml-org/gemma-3-4b-it-qat-GGUF(元: Google) |
| Whisper large-v3-turbo(ggml、q5_0 または f16) | 音声認識(精度重視) | MIT(OpenAI) | huggingface.co/ggerganov/whisper.cpp |
| SenseVoice-Small(sherpa-onnx の int8 版) | 既定の音声認識(速い耳) | **FunASR Model Open Source License v1.1**(https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)。出典の表示・モデル名の保持が条件。明示的な非商用の制限は無いと読めるが、SPDX の識別子が無いため **要確認** | github.com/k2-fsa/sherpa-onnx/releases(元: FunAudioLLM / Alibaba) |
| Kokoro-82M v1.0(sherpa-onnx の多言語版) | 読み上げ(英・中・日など) | Apache-2.0。アーカイブ内に eSpeak NG のデータ(GPL-3.0-or-later)が含まれるが、sherpa-onnx 2.0 は使わず、Sokutsu も eSpeak NG を同梱しない。学習データの謝辞: Koniwa tnc(CC BY 3.0)、SIWIS(CC BY 4.0)。内包の辞書: misaki(Apache-2.0)、pypinyin(MIT)、CppJieba(MIT) | github.com/k2-fsa/sherpa-onnx/releases(元: hexgrad) |
| 3D-Speaker CAM++(zh+en 学習の話者埋め込み) | 話者の見分け | Apache-2.0 | github.com/k2-fsa/sherpa-onnx/releases(元: Alibaba 3D-Speaker) |
| Silero VAD | 発話区間の検出 | MIT | github.com/snakers4/silero-vad |
| 翻訳に使う任意のモデル(例: Qwen 系、リモートサーバー側) | 「リモートGPU」の翻訳 | 使うモデルごとに異なる(Qwen 系は多くが Apache-2.0 だが、モデルにより異なるため **要確認**) | 利用者が選ぶ |
| COMET `Unbabel/wmt22-comet-da`(評価のみ) | 翻訳品質の評価 | **要確認**(配布元のモデルカードで確認) | huggingface.co/Unbabel/wmt22-comet-da |

## C. 外部アプリ・音声(同梱しない。利用者が自分で入れる)

| 名前 | 備考 |
|---|---|
| VOICEVOX ENGINE と各キャラクターの音声ライブラリ | 日本語の声に任意で使用(`scripts/voicevox_local.sh`)。**キャラクターごとに利用規約・クレジット表記の条件が異なります**(公開する動画・音声に VOICEVOX の表記が必要なものがあります)。利用者が https://voicevox.hiroshiba.jp/ の規約を確認してください。Sokutsu は VOICEVOX を同梱しません。**要確認**: 声の ID(例: `voicevox:13`)の既定が指す声の規約 |
| LM Studio | 翻訳サーバーとして任意で使用。同梱しません。LM Studio 自身の規約に従ってください |
| macOS の音声(`say`)・Apple のフレームワーク | OS に含まれるもの。Apple の規約に従います |

## D. データセット(評価。リポジトリには文を一部含む)

| データ | ライセンス | 備考 |
|---|---|---|
| FLEURS(Google) | CC BY 4.0 | `eval/` の評価セットは、FLEURS の並行文を使うスクリプトと、作った文から成ります。FLEURS の音声・全文は含みません(`scripts/eval_conv.py` で作り直します) |
| AMI Meeting Corpus | CC BY 4.0 | 英語の実会議の評価(`scripts/eval_live.py`)に使用。音声・書き起こしは含みません |

## E. このリポジトリ内の素材

| 素材 | 備考 |
|---|---|
| `src/tsuyaku/data/termbase.json`(内蔵の用語集) | プロジェクトで作成した日英中韓の対訳表。外部の用語集の丸写しではない想定ですが、由来の記録を持っていないため **要確認** |
| `src/tsuyaku/assets/`・`webui/logo.svg`(アイコン・ロゴ) | プロジェクトで作成。第三者の素材は使っていない想定(**要確認**) |
| `eval/` の文(用語テスト、対話、断片など) | プロジェクトで作った文。人の発言ではありません |
