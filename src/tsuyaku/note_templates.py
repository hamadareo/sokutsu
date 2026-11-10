"""会議メモのテンプレート(11/07 W1: Notion / tl;dv / Fireflies / Notta のテンプレートに相当)。

会議の種類ごとに、メモの重み(何を拾うか)と、足す節(sections)を変える。
「自由」(free)は今までのメモそのもの(指示文に何も足さない。AMI の評価で品質を確かめた既定)。
ほかの種類は、共通の項目(要約・話題・決定・やること・質問)はそのままに、種類ごとの観点の指示と
節を足す。節の中身は文字列の並び(各項目の終わりに根拠の行番号 [#12])。

議題(agenda)と会議中の自分のメモ(memos)もここで指示文にする(W3 が記録に書く。データの約束:
設計メモ(公開版には含めない))。
"""
from __future__ import annotations

EN = {"ja": "Japanese", "en": "English", "zh": "Simplified Chinese", "ko": "Korean"}

# 種類: 名前(4言語)・観点(英語の指示)・節(key → 4言語の見出し)。並びは画面の並び。
TEMPLATES: dict = {
    "free": {
        "label": {"ja": "自由", "en": "General", "zh": "通用", "ko": "자유"},
        "focus": "",
        "sections": {},
    },
    "regular": {
        "label": {"ja": "定例", "en": "Team meeting", "zh": "例会", "ko": "정기 회의"},
        "focus": ("This is a recurring team meeting (status / check-in). Emphasize the progress of "
                  "each ongoing item and who reported it, blockers and issues, decisions, and action "
                  "items with owners and due dates."),
        "sections": {
            "progress": {"ja": "進捗", "en": "Progress updates", "zh": "进展", "ko": "진행 상황"},
            "blockers": {"ja": "課題・困りごと", "en": "Blockers and issues", "zh": "问题与阻碍", "ko": "과제·어려움"},
        },
    },
    "1on1": {
        "label": {"ja": "1on1", "en": "1:1", "zh": "一对一", "ko": "1:1 면담"},
        "focus": ("This is a one-on-one between two people (for example a manager or mentor and a team "
                  "member). Emphasize the member's updates and achievements, worries and requests, "
                  "feedback given in either direction, growth and career topics, and the agreed next "
                  "steps. Be respectful and factual; do not judge the person beyond what was said."),
        "sections": {
            "updates": {"ja": "近況・成果", "en": "Updates and wins", "zh": "近况与成果", "ko": "근황·성과"},
            "concerns": {"ja": "悩み・相談", "en": "Concerns and requests", "zh": "烦恼与请求", "ko": "고민·상담"},
            "feedback": {"ja": "フィードバック", "en": "Feedback", "zh": "反馈", "ko": "피드백"},
            "growth": {"ja": "成長・キャリア", "en": "Growth and career", "zh": "成长与职业", "ko": "성장·커리어"},
        },
    },
    "sales": {
        "label": {"ja": "商談", "en": "Sales call", "zh": "商务洽谈", "ko": "영업 미팅"},
        "focus": ("This is a sales or business meeting with a customer or partner. Emphasize the "
                  "customer's needs and pain points, what was proposed, objections and concerns, "
                  "budget, timeline and decision makers, and the next steps."),
        "sections": {
            "needs": {"ja": "相手の課題・要望", "en": "Customer needs", "zh": "客户需求", "ko": "고객의 과제·요구"},
            "proposal": {"ja": "提案した内容", "en": "What we proposed", "zh": "提案内容", "ko": "제안 내용"},
            "objections": {"ja": "懸念・反論", "en": "Objections and concerns", "zh": "顾虑与异议", "ko": "우려·반론"},
            "deal": {"ja": "予算・時期・決裁", "en": "Budget, timeline, decision makers", "zh": "预算·时间·决策人",
                     "ko": "예산·시기·결재"},
        },
    },
    "interview": {
        "label": {"ja": "面接", "en": "Interview", "zh": "面试", "ko": "면접"},
        "focus": ("This is a job interview. Summarize the candidate's background, experience and their "
                  "answer to each question factually, as said. Note strengths and points to confirm "
                  "only when grounded in what was said. Never infer or mention age, gender, "
                  "nationality, family, health, religion or other personal attributes, and do not "
                  "make a hiring recommendation."),
        "sections": {
            "background": {"ja": "経歴・経験", "en": "Background and experience", "zh": "经历与经验", "ko": "경력·경험"},
            "qa": {"ja": "質問と回答", "en": "Questions and answers", "zh": "问答", "ko": "질문과 답변"},
            "strengths": {"ja": "強み", "en": "Strengths", "zh": "优势", "ko": "강점"},
            "concerns": {"ja": "確かめたい点", "en": "Points to confirm", "zh": "待确认的点", "ko": "확인할 점"},
        },
    },
    "lecture": {
        "label": {"ja": "授業・講義", "en": "Lecture", "zh": "课程·讲座", "ko": "수업·강의"},
        "focus": ("This is a lecture or class. Write notes a student can study from: key concepts and "
                  "definitions, the explanations and examples in order, important facts or formulas, "
                  "and hints about exams. todos = assignments, homework or things students should do."),
        "sections": {
            "concepts": {"ja": "重要な概念・用語", "en": "Key concepts and terms", "zh": "重要概念与术语", "ko": "핵심 개념·용어"},
            "examples": {"ja": "説明と例", "en": "Explanations and examples", "zh": "讲解与例子", "ko": "설명과 예시"},
            "exam": {"ja": "試験・大事な点", "en": "Exam and key points", "zh": "考试与重点", "ko": "시험·중요한 점"},
        },
    },
    "brainstorm": {
        "label": {"ja": "ブレスト", "en": "Brainstorm", "zh": "头脑风暴", "ko": "브레인스토밍"},
        "focus": ("This is a brainstorming session. Capture every distinct idea and who proposed it, "
                  "group the ideas into themes, note which ideas got support or were shortlisted, "
                  "and the next steps."),
        "sections": {
            "ideas": {"ja": "出たアイデア", "en": "Ideas", "zh": "提出的想法", "ko": "나온 아이디어"},
            "themes": {"ja": "テーマ・まとまり", "en": "Themes", "zh": "主题归类", "ko": "주제·묶음"},
            "shortlist": {"ja": "有望な案", "en": "Shortlist", "zh": "看好的方案", "ko": "유망한 안"},
        },
    },
    "qa": {
        "label": {"ja": "学会の質疑", "en": "Conference Q&A", "zh": "学术问答", "ko": "학회 질의응답"},
        "focus": ("This is the question-and-answer session after an academic presentation. For each "
                  "question write who asked (name or affiliation if said), the question and the "
                  "presenter's answer, as one item \"Q: … / A: …\". Also note suggestions and "
                  "criticisms. questions = questions that were not fully answered. todos = things the "
                  "presenter should check, add or revise."),
        "sections": {
            "qa": {"ja": "質問と回答", "en": "Questions and answers", "zh": "问题与回答", "ko": "질문과 답변"},
            "suggestions": {"ja": "指摘・提案", "en": "Suggestions and criticism", "zh": "意见与建议", "ko": "지적·제안"},
        },
    },
}


def normalize(key) -> str:
    return key if isinstance(key, str) and key in TEMPLATES else "free"


def label(key: str, lang: str) -> str:
    lb = TEMPLATES[normalize(key)]["label"]
    return lb.get(lang) or lb["en"]


def section_keys(key: str) -> list[str]:
    return list(TEMPLATES[normalize(key)]["sections"])


def section_title(key: str, sec: str, lang: str) -> str:
    t = TEMPLATES[normalize(key)]["sections"].get(sec) or {}
    return t.get(lang) or t.get("en") or sec


def schema_extra(key: str) -> str:
    """JSON の形に足す部分(自由なら空)。"""
    ks = section_keys(key)
    if not ks:
        return ""
    return ', "sections": {' + ", ".join(f'"{k}": [str]' for k in ks) + "}"


def context(key: str, agenda: str, memos: list, lang: str, stage: str = "full") -> str:
    """指示文に足す段落。stage: full(1回で作る)/ part(区切りごとの抜き出し)/ merge(まとめ)。
    自由・議題なし・自分のメモなしなら空(今までの指示文そのまま)。"""
    out = []
    key = normalize(key)
    L = EN.get(lang, "Japanese")
    t = TEMPLATES[key]
    if t["focus"]:
        secs = "; ".join(f"{k} = {section_title(key, k, 'en')}" for k in t["sections"])
        out.append(f"Meeting type: {label(key, 'en')}. {t['focus']}\n"
                   f"Also fill \"sections\" ({secs}): items as short sentences in {L}. "
                   "Empty list if nothing fits.")
    agenda = (agenda or "").strip()
    if agenda:
        if stage == "part":
            how = ("When this part discusses an agenda item, use that item (in "
                   f"{L}) as the topic title. Skip agenda items not discussed in this part.")
        else:
            how = (f"Organize topics along the agenda: one topic per agenda item, in agenda order, titled "
                   f"with the agenda item (in {L}); put other subjects in extra topics after them. If an "
                   "agenda item was not discussed, keep it as a topic with no points.")
        out.append(f"Agenda written before the meeting:\n{agenda}\n{how}")
    ms = [m for m in (memos or []) if isinstance(m, dict) and str(m.get("text") or "").strip()]
    if ms:
        rows = "\n".join(f"- {m.get('t', '')} {str(m['text']).strip()}" for m in ms[:40])
        out.append("The user wrote these notes during the meeting (time, text). They mark what the user "
                   f"found important: make sure the notes cover these points.\n{rows}")
    return "\n\n".join(out)
