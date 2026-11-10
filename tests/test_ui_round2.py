"""UI 改善ラウンド2の構造テスト(index.html を文字列/正規表現で検査する)。

対象:
1. 起動画面のペアボタンが相手言語の native 表記
2. 起動中(work)でまだ1文もないときの進捗メッセージ
3. テーマ入力欄で Enter → #go を押す
4. Esc: モーダル→閉じる、busy→interrupt、何も無いときだけ stop
5. 読み上げ中止のヒント文言
6. 履歴レールの .item クリックで大字幕に引き出す
7. エラー #err は 12秒で自動消える
8. #stopSpeak と Space ハンドラの退行防止
"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "src" / "tsuyaku"
        / "webui" / "index.html").read_text()


def _slice(start_marker, end_marker):
    """start_marker の位置から end_marker 直前までのテキストを返す(無ければ失敗)。"""
    i = HTML.index(start_marker)
    j = HTML.index(end_marker, i)
    return HTML[i:j]


class TestUIRound2(unittest.TestCase):
    def test_pair_buttons_use_native_names(self):
        labels = [
            "日本語 ⇄ English",
            "日本語 ⇄ 中文",
            "日本語 ⇄ 한국어",
            "English ⇄ 中文",
            "English ⇄ 한국어",
            "中文 ⇄ 한국어",
        ]
        for label in labels:
            # 11/06: 左右の言語を色分けするため span に分けた(表記は各自の言語のまま)
            a, b = label.split(" ⇄ ")
            html = (f'<span class="pa">{a}</span><span class="px">⇄</span>'
                    f'<span class="pb">{b}</span>')
            self.assertIn(html, HTML, f"ペアボタン表記 {label!r} が無い")
        for native in ("English", "中文", "한국어"):
            self.assertIn(native, HTML)

    def test_no_simplified_shiting(self):
        # 简体の「试听」は日本語の画面に混入してはいけない。日本語の「試聴」は入っている。
        # 11/06: 画面の言語に中文を足したので、中文の辞書(I18N-BEGIN〜END)の中だけは許す。
        i, j = HTML.index("/*I18N-BEGIN*/"), HTML.index("/*I18N-END*/")
        self.assertNotIn("试听", HTML[:i] + HTML[j:])
        self.assertIn("試聴", HTML)

    def test_startup_progress_message_in_status(self):
        body = _slice("status(state,text){", "level(v){")
        self.assertIn("1〜2分", body)
        self.assertIn('"起動中":"サーバーを起動しています… 初回はモデルの読み込みに1〜2分"', body)
        self.assertIn('"接続中":"翻訳サーバーを繋いでいます…"', body)
        self.assertIn("モデル読み込み中", body)

    def test_enter_in_topic_clicks_go(self):
        body = _slice('getElementById("topic").addEventListener("keydown"',
                      'document.addEventListener("keydown"')
        self.assertIn('"Enter"', body)
        self.assertIn('getElementById("go")', body)
        self.assertIn(".click()", body)

    def test_esc_interrupts_when_busy(self):
        body = _slice('if(e.key==="Escape"){', "});")
        self.assertIn("TSUYAKU.busy()", body)
        self.assertIn("TSUYAKU.interrupt()", body)
        # busy の interrupt より後に stop() が来る(読み上げ中の Esc が即終了しない)
        self.assertLess(body.index("interrupt()"), body.index("api.stop()"))

    def test_error_auto_clears_after_12s(self):
        self.assertIn("_errT=setTimeout", HTML)
        self.assertIn(",12000)", HTML)

    def test_log_click_pulls_item_to_stage(self):
        body = _slice('getElementById("log").addEventListener("click"',
                      'getElementById("stopSpeak")')
        self.assertIn('closest(".item")', body)
        self.assertIn('.querySelector(".ti")', body)
        self.assertIn('.querySelector(".oi")', body)
        self.assertIn('#orig', body)
        self.assertIn('#trans', body)

    def test_stop_speak_button_and_space_still_alive(self):
        # 既存機能の退行防止: #stopSpeak ボタンと Space ハンドラが生きている
        self.assertIn('id="stopSpeak"', HTML)
        self.assertIn('getElementById("stopSpeak").addEventListener("click"', HTML)
        space = _slice('(e.key===" "||e.code==="Space")', "});")
        self.assertIn("TSUYAKU.busy()", space)
        self.assertIn("TSUYAKU.interrupt()", space)

    def test_hint_mentions_esc_and_space(self):
        self.assertIn("Space か字幕クリックで読み上げ中止", HTML)
        # 11/05: 画面上の「終了」ボタンを追加したので、案内もそれを先に挙げる
        self.assertIn("終了は右上の「終了」か Esc", HTML)

    def test_short_window_compaction(self):
        # 縦に短い窓(13インチや分割画面)では起動画面を詰めて CTA を画面内に収める
        self.assertIn("@media (max-height:860px)", HTML)
        block = _slice("@media (max-height:860px){", "\n}\n")
        for sel in ("#start .sc", "#start .mark", "#start .tag",
                    ".pair", ".vrow", "#topic", "#go"):
            self.assertIn(sel, block, f"短窓詰め込み {sel} が無い")

    def test_item_hover_style(self):
        self.assertIn(".item:hover{background:var(--ink-3)}", HTML)
        self.assertIn(".item{", HTML)
        self.assertIn("cursor:pointer", _slice(".item{", "\n"))


class TestGuessAndPillars(unittest.TestCase):
    """11/06 デザイン一新(二本の柱)と予想訳(partial の第4引数 guess)の契約。"""

    def test_partial_accepts_guess_flag(self):
        body = _slice("partial(orig,trans,lang,guess,spk){", "error(title,body){")
        # 予想訳は #stage.guess で表す(淡い字+点線の下線)。無い/false なら従来の途中訳
        self.assertIn('stage.classList.toggle("guess",!!guess)', body)
        # 直前の確定の差し替え(120ms後)が予想訳を上書きしない
        self.assertIn("clearTimeout(this._fadeT)", body)

    def test_guess_style_keeps_layout(self):
        # 予想訳と確定で字の大きさ・太さは同じ(色と下線だけ変える)= 確定の瞬間に文字が動かない
        rule = _slice("#stage.guess #trans{", "}")
        for prop in ("font-size", "font-weight", "letter-spacing", "line-height"):
            self.assertNotIn(prop, rule)
        self.assertIn("text-decoration-color", rule)

    def test_confirm_same_text_does_not_swap(self):
        body = _slice("utterance(u){", "partial(orig,trans,lang,guess,spk){")
        # 予想訳がそのまま確定したら fade で差し替えない
        self.assertIn("const same=", body)
        self.assertIn('stage.classList.remove("guess"', body)

    def test_pillars_exist_and_follow_pair(self):
        self.assertIn('id="pA"', HTML)
        self.assertIn('id="pB"', HTML)
        # ペアを選ぶと柱の言語名が変わる
        # 11/06 会議メモ: 1言語の札も同じ処理(selPair)にまとめた
        self.assertIn('pairs.forEach(b=>b.addEventListener("click",()=>TSUYAKU.selPair(b.dataset.pair)))', HTML)
        sel = _slice("selPair(p){", "langs(){")
        self.assertIn("this.pillars()", sel)
        # 読み上げの届け先は status("speak", "読み上げ中 (英語)") の言語名から決める
        st = _slice("status(state,text){", "level(v){")
        self.assertIn('state==="speak"', st)

    def test_latency_view_wired(self):
        # __main__.py は utterance に id/lat を載せ、声の鳴り始めを TSUYAKU.voice({id,lat}) で送る
        self.assertIn("voice(d){", HTML)
        body = _slice("voice(d){", "_same(a,b)")
        self.assertIn("d.id", body)
        self.assertIn("dataset.uid", HTML)
        self.assertIn('id="latNow"', HTML)


class TestOpeningAndLatency(unittest.TestCase):
    """11/06「最初のオープニングからすごく凝ったソフトに」「応答速度を可視化」。"""

    def test_opening_is_short_skippable_and_respects_reduced_motion(self):
        # 描画前に印を付ける。動きを減らす設定なら付けない(静的な起動画面)
        self.assertIn('matchMedia("(prefers-reduced-motion: reduce)").matches)document.body.classList.add("opening")', HTML)
        # 1.4秒で必ず終わる(開いてから操作できるまで1.5秒以内)。押せばその場で終わる
        self.assertIn("TSUYAKU._openT=setTimeout(end,1400)", HTML)
        self.assertIn('window.addEventListener("pointerdown",end,true)', HTML)
        self.assertIn('window.addEventListener("keydown",end,true)', HTML)
        # 飛ばすための Esc でアプリを閉じない
        seg = _slice("const end=e=>{", "TSUYAKU._openT=")
        self.assertIn('e.key==="Escape"', seg)
        self.assertIn("stopPropagation()", seg)

    def test_boot_animation_follows_real_progress(self):
        body = _slice("boot(d){", "bootHide(){")
        # 柱の満ちは boot の step に合わせるだけ(演出で起動を遅らせない)
        self.assertIn('setProperty("--bp"', body)
        # 完了から会話画面まで +0.5秒以内
        self.assertIn("setTimeout(()=>this.bootHide(),500)", body)

    def test_latency_breakdown(self):
        self.assertIn('id="latModal"', HTML)
        self.assertIn("latOpen(){", HTML)
        # 直近20発話・区間は 聞き取り/翻訳/声 の3つ
        body = _slice("latOpen(){", "voice(d){")
        self.assertIn("slice(-20)", body)
        for lab in ("聞き取り", "翻訳", "声の準備"):
            self.assertIn(lab, body)
        # 下の帯は直近5発話の中央値、押すと内訳
        self.assertIn("this._lats.slice(-5)", HTML)
        self.assertIn('getElementById("latNow").addEventListener("click"', HTML)
        # lat が null の発話は出さない
        self.assertIn("if(u.lat&&this._num(u.lat.caption)!=null)", HTML)

    def test_array_args_are_tolerated(self):
        # 旧い Python は partial/error を配列1個で呼んでいた
        self.assertIn("if(Array.isArray(orig))[orig,trans,lang,guess,spk]=orig;", HTML)
        self.assertIn("if(Array.isArray(title))[title,body]=title;", HTML)


class TestCaptionFit(unittest.TestCase):
    """11/06 実会話(長い英文→4行以上の和訳): 字数の目安ではなく実寸で縮める。文の途中は縮むだけ。"""

    def test_measured_fit_shrinks_only_within_sentence(self):
        self.assertIn("fitNow(){", HTML)
        body = _slice("fitNow(){", "},")
        self.assertIn("tr.offsetHeight>box.clientHeight", body)
        # 新しい文の始まり(確定の後の最初の partial)でだけ大きさを戻す
        part = _slice("partial(orig,trans,lang,guess,spk){", "error(title,body){")
        self.assertIn("if(!this._pending)this.fitReset();", part)
        # 予想訳がそのまま確定した時は戻さない(fitNow だけ)
        utt = _slice("if(same){", "}else{")
        self.assertNotIn("fitReset", utt)

    def test_caption_names_the_listener(self):
        # 訳の左の太い線 = 受け取る人の色
        self.assertIn('#stage[data-to="a"]{--dc:var(--a)}', HTML)
        self.assertIn("border-left:6px solid var(--dc,var(--line))", HTML)



class TestUiLanguage(unittest.TestCase):
    """11/06「グローバル対応」: 画面の文言を 日本語/English/中文/한국어 で切り替える(字幕そのものは対象外)。"""

    def test_switcher_and_default(self):
        self.assertIn('id="uiLang"', HTML)
        for lang in ("ja", "en", "zh", "ko"):
            self.assertIn(f'data-ui="{lang}"', HTML)
        # 既定は前回の選択 → OS の言語(navigator.language)
        self.assertIn('localStorage.getItem("tsuyaku.ui")', HTML)
        self.assertIn("navigator.language", HTML)

    def test_every_marked_text_has_all_translations(self):
        import re
        # 辞書は本体の塊 /*I18N-BEGIN*/…/*I18N-END*/ と、担当ごとの塊 /*W2-I18N-BEGIN*/… など
        spans = [(m.start(), HTML.index("I18N-END*/", m.end()))
                 for m in re.finditer(r"/\*(?:\w+-)?I18N-BEGIN\*/", HTML)]
        block = "".join(HTML[a:b] for a, b in spans)
        rest, last = "", 0
        for a, b in spans:
            rest += HTML[last:a]
            last = b
        rest += HTML[last:]
        # data-t の付いた要素の日本語は、3言語の辞書に必ずある
        keys = re.findall(r'<(?:div|span|button|h2|b)\s[^>]*?data-t(?=[\s>])[^>]*>(.*?)</(?:div|span|button|h2|b)>', rest)
        self.assertGreater(len(keys), 20)
        for k in keys:
            self.assertGreaterEqual(block.count('"' + k + '":'), 3, f"訳が足りない: {k}")

    def test_python_status_texts_are_translated(self):
        # Python は状態表示を日本語で送る。画面の言語で出し直す
        self.assertIn("stText(t){", HTML)
        self.assertIn("this.stText(text)", HTML)



class TestProjectorMode(unittest.TestCase):
    """11/06 明るい会場のプロジェクタ: 黒が白の15%まで浮く想定で、補助の文字も 3:1 以上にする投影モード。"""

    def test_projector_mode_tokens_and_toggle(self):
        self.assertIn("body.proj{--muted:#D5DBE0;--dim:#BCC5CD;--a:#FFD08F;--b:#9CEBE4;", HTML)
        self.assertIn('id="projBtn"', HTML)    # 会話中の上の帯
        self.assertIn('id="projStart"', HTML)  # 起動画面の右上
        self.assertIn('localStorage.getItem("tsuyaku.proj")', HTML)
        # 柱の canvas も切替に合わせて色を読み直す
        body = _slice("setProj(on,save){", "sigStop(){")
        self.assertIn("this.sigColors()", body)

    def test_projector_colors_reach_3_to_1(self):
        def lum(h):
            h = h.lstrip("#")
            c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
            f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
            return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])
        wash = lambda l: l * 0.8 + 0.15   # 投影: 白は80%に落ち、黒は白の15%まで浮く
        bg = wash(lum("#0D1217"))
        for col in ("#D5DBE0", "#BCC5CD", "#FFD08F", "#9CEBE4"):
            self.assertGreaterEqual((wash(lum(col)) + 0.05) / (bg + 0.05), 3.0, col)



class TestTurnTaking(unittest.TestCase):
    """11/06 読み上げ中はマイクを閉じている。相手が途中から話すと捨てられるので、順番を目で見せる。"""

    def test_speaking_band_says_mic_is_paused(self):
        self.assertIn('id="micNote"', HTML)
        self.assertIn("マイク休止中 — 読み上げが終わったら話してください", HTML)
        self.assertIn("body.speaking #status{background:var(--cur)", HTML)

    def test_listen_after_speak_marks_your_turn(self):
        st = _slice("status(state,text){", "// ---------- 会話開始時の読み込みバー")
        self.assertIn('was==="speak"&&state==="listen"', st)
        self.assertIn("this.yourTurn(", st)
        self.assertIn("聞いています — どうぞ", HTML)
        self.assertIn(".pillar.drain .pfill", HTML)   # 満ちていた柱の縞が下へ引く

    def test_app_grid_does_not_grow_with_status_text(self):
        # 帯の長い文言で #app の列が窓より広がり、記録や終了ボタンが柱の下に隠れた(11/06)
        self.assertIn("grid-template-columns:minmax(0,1fr)}", HTML)


class TestGuessHold(unittest.TestCase):
    """先出しの予想訳(このMacの耳)→ 原文が差し替わる → 次の予想訳。「訳→…→訳」で点滅させない。"""

    def test_dots_do_not_blank_a_shown_guess(self):
        body = _slice("partial(orig,trans,lang,guess,spk){", "unhold(){")
        self.assertIn('t==="…"&&stage.classList.contains("guess")', body)
        self.assertIn('stage.classList.add("held")', body)
        self.assertIn(",700)", body)
        utt = _slice("utterance(u){", "partial(orig,trans,lang,guess,spk){")
        self.assertIn("this.unhold()", utt)



class TestFirstUseHelp(unittest.TestCase):
    """11/06 初めて使う人(海外の相手を含む)がすぐ分かるように: 設定の一言と操作の早見。"""

    def test_setting_notes_follow_selection(self):
        self.assertIn('id="engineNote"', HTML)
        self.assertIn('id="noiseNote"', HTML)
        body = _slice("segNotes(){", "engineMark(eng){")
        self.assertIn("this.ENG_NOTE", body)
        self.assertIn("this.NOISE_NOTE", body)
        i, j = HTML.index("/*I18N-BEGIN*/"), HTML.index("/*I18N-END*/")
        for k in ("リモートの GPU サーバーで翻訳します。速く正確ですが、ネットワークが要ります",
                  "周りの話し声や雑音を拾いにくくします。少し大きめに、はっきり話してください"):
            self.assertEqual(HTML[i:j].count('"' + k + '":'), 3, k)

    def test_quick_help_opens_with_question_key(self):
        self.assertIn('id="helpModal"', HTML)
        self.assertIn('id="helpBtn"', HTML)
        self.assertIn('id="helpStart"', HTML)
        self.assertIn('e.key==="?"', HTML)
        esc = _slice('if(e.key==="Escape"){', "});")
        self.assertIn("closeHelp()", esc)



class TestNetIndicator(unittest.TestCase):
    """11/06 回線の混み具合: Python が ("net", {slow, net_s}) を送り TSUYAKU.net を呼ぶ。"""

    def test_net_shows_only_while_slow(self):
        self.assertIn("net(d){", HTML)
        body = _slice("net(d){", "engineInfo(d){")
        self.assertIn("t.hidden=!on", body)
        self.assertIn("d.slow", body)
        self.assertIn("net_s", body)
        # 会話の終わり(reset)で消す
        r = _slice("reset(){", "document.addEventListener(\"DOMContentLoaded\"")
        self.assertIn("this.net({slow:false})", r)

    def test_net_tag_in_status_bar_and_help(self):
        self.assertIn('<span id="netTag" hidden>', HTML)
        self.assertLess(HTML.index('id="netTag"'), HTML.index('id="modelTag"'))  # 処理場所の表示の隣
        self.assertIn("回線が混んでいます — 予想訳を先に出しています", HTML)
        i, j = HTML.index("/*I18N-BEGIN*/"), HTML.index("/*I18N-END*/")
        self.assertEqual(HTML[i:j].count('"回線が混んでいます — 予想訳を先に出しています":'), 3)
        # 早見の「画面の見方」にも1行
        h = _slice('id="helpModal"', 'id="helpClose"')
        self.assertIn("回線が混んでいます", h)



class TestWebKitLook(unittest.TestCase):
    """本番は pywebview の WKWebView。Chrome とだけ見比べて見落とした差を固定する(11/06 WebKit で実描画して確認)。"""

    def test_speed_slider_is_custom_drawn(self):
        # WKWebView では accent-color が効かず、白い標準のスライダーが暗い画面から浮いていた
        self.assertIn("input[type=range].vspd::-webkit-slider-thumb", HTML)
        self.assertIn("input[type=range].vspd::-webkit-slider-runnable-track", HTML)
        body = _slice("buildVoiceRows(sel){", "setTts(lang,voice,speed){")
        self.assertIn('rng.style.setProperty("--p"', body)



class TestTranscriptCopy(unittest.TestCase):
    """11/06 学会の会話のあとで見返す・貼り付ける: 記録レールの「記録をコピー」。"""

    def test_copy_button_and_text(self):
        self.assertIn('id="copyLog"', HTML)
        body = _slice("transcriptText(){", "copyTranscript(){")
        for part in ('.querySelector(".time")', '.querySelector(".oi")', '.querySelector(".ti")', "this.ln(lang)"):
            self.assertIn(part, body)
        # 先頭に今日の会話のまとめ(発話数・中央値・予想訳が先に出た回数)
        self.assertIn("予想訳が先に出た {g}回", body)

    def test_copy_works_without_clipboard_api(self):
        # WKWebView の file:// で navigator.clipboard が無い/拒まれた時は textarea + execCommand
        body = _slice("copyTranscript(){", "unhold(){")
        self.assertIn("navigator.clipboard", body)
        self.assertIn('document.execCommand("copy")', body)

    def test_open_log_only_if_python_has_it(self):
        self.assertIn('id="openLog"', HTML)
        self.assertIn('typeof _a.open_log==="function"', HTML)



class TestMeetingNotesUI(unittest.TestCase):
    """11/06 会議メモ(設計 設計メモ(公開版には含めない))の画面側。"""

    def test_use_mode_selector(self):
        for m in ("interpret", "notes", "both"):
            self.assertIn(f'data-mode="{m}"', HTML)
        # 会議メモだけ は1言語も選べる
        for p in ("ja", "en", "zh", "ko"):
            self.assertIn(f'data-pair="{p}"', HTML)
        # 会議メモだけ では読み上げ・声の欄を隠す
        self.assertIn("body.m-notes .vfield,body.m-notes .vofield", HTML)
        # 通訳だけ は今までどおり start(pair, topic)。ほかの使い方の時だけ mode を渡す(Python が古くても通訳は動く)
        self.assertIn("window.pywebview.api.start(TSUYAKU.pair,tp,TSUYAKU.mode)", HTML)
        self.assertIn('TSUYAKU.mode==="interpret"?window.pywebview.api.start(TSUYAKU.pair,tp)', HTML)

    def test_speakers_on_caption_and_rail(self):
        self.assertIn("partial(orig,trans,lang,guess,spk){", HTML)
        utt = _slice("utterance(u){", "partial(orig,trans,lang,guess,spk){")
        self.assertIn("this.showSpk(u.spk)", utt)
        self.assertIn("this.spkHtml(u.spk)", utt)
        self.assertIn("speakers(map){", HTML)       # 名前が分かった/直った時に印を書き直す
        self.assertIn('"話者{n}"', HTML)

    def test_meeting_records_use_bridge_api(self):
        for name in ("meetings_list", "meeting_get", "notes_make", "speaker_rename",
                     "todo_toggle", "notes_markdown", "notes_export"):
            self.assertIn(name, HTML, name)
        self.assertIn("notes(d){", HTML)             # TSUYAKU.notes({id, stage, pct, notes?, error?})
        self.assertIn('id="meet"', HTML)
        self.assertIn('id="notesBtn"', HTML)         # 会話中の「メモ」= 今の会話(live)のメモを作って開く
        self.assertIn('TSUYAKU.docOpen("live",null,true)', HTML)

    def test_notes_only_shows_original_big(self):
        utt = _slice("utterance(u){", "partial(orig,trans,lang,guess,spk){")
        self.assertIn('notesOnly?(u.orig||"")', utt)



class TestMeetingDocPolish(unittest.TestCase):
    """11/06「UIにはこだわって」: 会議メモの文書の読み心地と手応え。"""

    def test_crowd_strip_and_roles(self):
        self.assertIn("crowdHtml(parts,lines,names,pair){", HTML)   # 参加者の印と話した量の帯
        self.assertIn("docRoles(){", HTML)                           # participants[].role
        self.assertIn('if(spk==="all") return this.L("全員");', HTML)  # owner "all"

    def test_todo_progress_and_filter(self):
        self.assertIn('"{d}/{n} 完了"', HTML)
        self.assertIn("todoFilter(o){", HTML)
        self.assertIn('id="todoBar"', HTML)

    def test_long_transcript_stays_light(self):
        self.assertIn("_txLimit:300", HTML)
        self.assertIn("data-more", HTML)

    def test_pillars_hidden_in_meetings_and_mono_right_pillar_is_record(self):
        self.assertIn("#meet{position:fixed;inset:0;", HTML)
        self.assertIn('n.textContent=this.L("記録")', HTML)

    def test_progress_mentions_first_run_wakeup(self):
        self.assertIn('STEPS:["read","chunk","merge"]', HTML)
        self.assertIn("1〜2分かかることがあります", HTML)



class TestRelabel(unittest.TestCase):
    """11/06 dd4d15d: 会話中の話者の付け直し TSUYAKU.relabel({発話id: "S2"})。"""

    def test_relabel_updates_rail_and_caption(self):
        body = _slice("relabel(map){", "speakers(map){")
        self.assertIn("x.dataset.uid===String(uid)", body)   # 記録の行
        self.assertIn("this._shownUid", body)                # 今の字幕
        self.assertIn("this.spkHtml(spk)", body)             # 名前の対応(spkNames)はそのまま効く


if __name__ == "__main__":
    unittest.main()
