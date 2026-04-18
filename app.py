import streamlit as st
import os
import time
from agent.react_agent import ReactAgent
from agent.psych_consult_agent import PsychConsultAgent
from agent.tools.agent_tools import rag as rag_service
from psych.graphrag_service import PsychGraphRagService
from utils.bootstrap import validate_runtime
from utils.chat_session_store import (
    create_session,
    delete_session,
    load_sessions,
    save_sessions,
    sort_sessions,
    update_session_messages,
    upsert_session,
)
from utils.config_handler import agent_conf
from utils.file_handler import clean_text, pdf_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.upload_memory_store import add_upload_memory, get_upload_memory
import re


st.set_page_config(
    page_title="医疗分诊对话助手",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

    :root {
        --brand-ink: #19324a;
        --brand-cyan: #3da9a3;
        --brand-gold: #f2c26b;
        --bg-soft: #f5f8fa;
        --surface: #ffffff;
        --surface-2: #ecf3f7;
        --line: #d9e3ea;
        --text-main: #13293d;
        --text-muted: #2f4d63;
    }

    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', sans-serif;
        color: var(--text-main);
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    .stApp {
        background:
            radial-gradient(1200px 420px at 100% -10%, rgba(61,169,163,0.22), transparent 70%),
            radial-gradient(900px 400px at -10% 5%, rgba(242,194,107,0.24), transparent 65%),
            linear-gradient(180deg, #f8fbfd 0%, #f2f6f9 100%);
    }

    .main .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 6.5rem;
    }

    .hero-wrap {
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1rem 1.25rem;
        background: linear-gradient(150deg, rgba(255,255,255,0.94), rgba(236,243,247,0.92));
        box-shadow: 0 12px 30px rgba(25,50,74,0.08);
        animation: riseIn 0.45s ease-out;
    }

    .hero-title {
        margin: 0;
        color: var(--brand-ink);
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }

    .hero-sub {
        margin: 0.35rem 0 0;
        color: var(--text-muted);
        font-size: 1.06rem;
        font-weight: 600;
    }

    .stat {
        margin-top: 0.9rem;
        display: inline-block;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        border: 1px solid #c5d7e3;
        background: #f8fcff;
        color: #1f3d54;
        font-size: 0.9rem;
        font-weight: 700;
        margin-right: 0.45rem;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 7px 18px rgba(15,43,66,0.07);
        animation: riseIn 0.35s ease-out;
    }

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        font-size: 1.08rem;
        line-height: 1.86;
        color: var(--text-main);
        font-weight: 500;
    }

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
        color: #143248;
        letter-spacing: 0.2px;
        font-weight: 800;
    }

    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        transform: scale(1.08);
    }

    [data-testid="stChatInput"] {
        position: fixed;
        left: 0;
        right: 0;
        margin: 0 auto;
        transform: none;
        bottom: 1rem;
        width: calc(100% - 1.6rem);
        max-width: 940px;
        background: rgba(255,255,255,0.93);
        border: 1px solid var(--line);
        border-radius: 18px;
        box-shadow: 0 16px 34px rgba(15,43,66,0.13);
        padding: 0.22rem 0.72rem;
        backdrop-filter: blur(8px);
        box-sizing: border-box;
        overflow: hidden;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] textarea {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 1.08rem;
        font-weight: 500;
        color: var(--text-main);
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="column"] .stButton button {
        width: 100%;
        border-radius: 999px;
        border: 1px solid #c0d4e2;
        background: #f7fbfe;
        color: #1e3f57;
        font-weight: 700;
        font-size: 0.98rem;
        transition: all 0.18s ease;
    }

    div[data-testid="stHorizontalBlock"] div[data-testid="column"] .stButton button:hover {
        background: #e7f4f3;
        border-color: #7fbcb7;
        color: #1f4541;
    }

    div[data-testid="stAlert"] p {
        color: #163349;
        font-size: 1rem;
        font-weight: 600;
    }

    .ref-wrap {
        margin-top: 0.85rem;
        padding-top: 0.8rem;
        border-top: 1px dashed #c9d8e3;
    }

    .ref-title {
        margin-bottom: 0.45rem;
        color: #38586f;
        font-size: 0.88rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .ref-chip {
        display: inline-block;
        margin: 0 0.45rem 0.45rem 0;
        padding: 0.32rem 0.62rem;
        border-radius: 999px;
        border: 1px solid #c6d9e6;
        background: #f4fafc;
        color: #214158;
        font-size: 0.86rem;
        font-weight: 700;
        line-height: 1.3;
    }

    .ref-preview {
        margin-top: 0.2rem;
        color: #28465b;
        font-size: 0.96rem;
        line-height: 1.75;
    }

    @keyframes riseIn {
        from {
            transform: translateY(8px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 7.2rem;
        }

        .hero-title {
            font-size: 1.58rem;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            font-size: 1rem;
            line-height: 1.78;
        }

        [data-testid="stChatInput"] {
            width: calc(100% - 0.9rem);
            max-width: none;
            bottom: 0.4rem;
            border-radius: 14px;
            padding: 0.2rem 0.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

runtime_issues = validate_runtime()
if runtime_issues:
    for issue in runtime_issues:
        st.error(issue)
    st.stop()

# Agent 实例只初始化一次，避免每次重跑页面都重新构建模型和工具。
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "psych_agent" not in st.session_state:
    st.session_state["psych_agent"] = PsychConsultAgent()

if "psych_rag_service" not in st.session_state:
    st.session_state["psych_rag_service"] = PsychGraphRagService()

# sessions 存的是全部历史会话；current_session_id 指向当前打开的那一个。
if "sessions" not in st.session_state:
    sessions = sort_sessions(load_sessions())
    if not sessions:
        sessions = [create_session()]
        save_sessions(sessions)
    st.session_state["sessions"] = sessions

if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = st.session_state["sessions"][0]["id"]

if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = ""

if "tone_mode" not in st.session_state:
    st.session_state["tone_mode"] = "温和"

if "output_mode" not in st.session_state:
    st.session_state["output_mode"] = "思考模式"

if "last_uploaded_sig" not in st.session_state:
    st.session_state["last_uploaded_sig"] = ""

if "module_mode" not in st.session_state:
    st.session_state["module_mode"] = "医疗分诊对话模块"


def get_current_session() -> dict:
    """根据 current_session_id 取当前会话；如果丢失则兜底创建一个新会话。"""
    current_session_id = st.session_state["current_session_id"]
    for session in st.session_state["sessions"]:
        if session["id"] == current_session_id:
            return session
    fallback = create_session()
    st.session_state["sessions"] = [fallback] + st.session_state["sessions"]
    st.session_state["current_session_id"] = fallback["id"]
    save_sessions(sort_sessions(st.session_state["sessions"]))
    return fallback


def persist_current_messages(messages: list[dict]) -> None:
    """把当前会话消息写回内存和本地文件。"""
    current = get_current_session()
    updated = update_session_messages(current, messages)
    st.session_state["sessions"] = sort_sessions(upsert_session(st.session_state["sessions"], updated))
    st.session_state["current_session_id"] = updated["id"]
    save_sessions(st.session_state["sessions"])


def switch_session(session_id: str) -> None:
    """切换当前会话，同时清空待发送的快捷问题。"""
    st.session_state["current_session_id"] = session_id
    st.session_state["pending_prompt"] = ""


def create_new_chat() -> None:
    """创建新会话并立即切过去。"""
    new_session = create_session()
    st.session_state["sessions"] = sort_sessions(upsert_session(st.session_state["sessions"], new_session))
    st.session_state["current_session_id"] = new_session["id"]
    st.session_state["pending_prompt"] = ""
    save_sessions(st.session_state["sessions"])


def delete_current_chat() -> None:
    """删除当前会话；如果删完为空，则自动补一个空会话。"""
    current_id = st.session_state["current_session_id"]
    sessions = delete_session(st.session_state["sessions"], current_id)
    if not sessions:
        sessions = [create_session()]
    sessions = sort_sessions(sessions)
    st.session_state["sessions"] = sessions
    st.session_state["current_session_id"] = sessions[0]["id"]
    st.session_state["pending_prompt"] = ""
    save_sessions(sessions)


def split_response_and_references(content: str) -> tuple[str, list[str]]:
    """
    把回答正文和“参考来源”拆开。

    RAG 最终返回的是一段完整文本，这里按约定格式拆分，
    方便前端把正文和引用来源分开展示。
    """
    if not content:
        return "", []

    match = re.search(r"\n参考来源：\s*\n(?P<refs>(?:- .+\n?)*)$", content.strip())
    if not match:
        return content.strip(), []

    body = content[: match.start()].strip()
    refs_block = match.group("refs")
    references = [line[2:].strip() for line in refs_block.splitlines() if line.startswith("- ")]
    return body, references


def render_references(references: list[str]):
    """把引用来源渲染成标签，并支持展开查看预览片段。"""
    if not references:
        return

    chips = "".join(f'<span class="ref-chip">{reference}</span>' for reference in references)
    st.markdown(
        f"""
        <div class="ref-wrap">
            <div class="ref-title">参考来源</div>
            <div>{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, reference in enumerate(references, start=1):
        with st.expander(f"查看片段 {index}: {reference}", expanded=False):
            st.caption("命中的本地知识片段预览")
            st.write(load_reference_preview(reference))


def parse_reference_label(reference: str) -> tuple[str, int | None]:
    """从“文件名 / 文件名 + 页码”格式中解析出来源和页码。"""
    match = re.match(r"^(?P<source>.+?)(?: 第(?P<page>\d+)页)?$", reference.strip())
    if not match:
        return reference.strip(), None
    source = match.group("source").strip()
    page = match.group("page")
    return source, int(page) - 1 if page else None


@st.cache_data(show_spinner=False)
def load_reference_preview(reference: str) -> str:
    """
    读取引用来源的本地预览片段。

    txt 直接读取文件开头片段；
    pdf 优先按页码读取对应页内容。
    """
    source, page = parse_reference_label(reference)
    abs_path = get_abs_path(f"data/{source}")
    if not abs_path or not source:
        return "未能解析参考来源。"

    try:
        if source.lower().endswith(".txt"):
            with open(abs_path, "r", encoding="utf-8") as f:
                preview = clean_text(f.read())[:420]
                return preview or "该文本来源没有可展示的预览内容。"
        if source.lower().endswith(".pdf"):
            docs = pdf_loader(abs_path)
            if page is not None and 0 <= page < len(docs):
                return clean_text(docs[page].page_content)[:420] or "该页没有可展示内容。"
            if docs:
                return clean_text(docs[0].page_content)[:420] or "PDF 没有可展示内容。"
            return "PDF 没有可展示内容。"
    except FileNotFoundError:
        return f"本地未找到来源文件：{source}"
    except Exception as e:
        logger.warning(f"加载参考片段失败: {reference}, error={str(e)}")
        return f"无法读取该来源的片段预览：{source}"

    return f"当前仅支持预览 txt/pdf 来源，文件：{source}"


def render_message(message: dict):
    """统一渲染一条消息，自动处理正文和引用来源。"""
    body, references = split_response_and_references(message["content"])
    st.markdown(body or message["content"], unsafe_allow_html=True)
    render_references(references)


def summarize_uploaded_text(file_name: str, data: bytes) -> str:
    """提取上传文件的可读摘要，写入会话记忆用于后续综合判断。"""
    lower_name = file_name.lower()
    if lower_name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
        return clean_text(text)[:800] or "文本文件内容为空。"
    return "该文件类型暂不支持内容摘要，已记录文件名。"


def _flow_template(module_mode: str) -> list[tuple[str, str]]:
    if module_mode == "医疗分诊对话模块":
        return [
            ("agent_boot", "Agent准备阶段"),
            ("collect", "收集用户信息"),
            ("sufficiency", "信息充足度判断"),
            ("severity", "严重程度评估"),
            ("route", "路由知识检索"),
            ("compose", "生成建议草案"),
            ("tone", "语气与风格调整"),
            ("tool_map", "工具调用：医院地图"),
            ("finish", "输出最终答复"),
        ]
    return [
        ("collect", "汇总心理困扰描述"),
        ("global", "Global Search 全局检索"),
        ("clarity", "模糊度判断与追问决策"),
        ("local", "Local Search 局部检索"),
        ("answer", "生成心理支持答复"),
    ]


def build_workflow_markdown(module_mode: str, events: list[dict]) -> str:
    """构建“工作流”段落，按节点逐行显示当前进度。"""
    if not events:
        return "开始 -> 节点迭代 -> 结束\n\n- 等待开始..."

    order_index = {key: idx for idx, (key, _) in enumerate(_flow_template(module_mode))}
    latest_by_key = {}
    seen_keys = []
    for event in events:
        key = event.get("key")
        if not key:
            continue
        if key not in latest_by_key:
            seen_keys.append(key)
        latest_by_key[key] = event

    seen_keys.sort(key=lambda k: order_index.get(k, 999))

    status_icon = {
        "pending": "⚪",
        "running": "🟡",
        "done": "🟢",
        "failed": "🔴",
    }
    status_text = {
        "pending": "待执行",
        "running": "进行中",
        "done": "已完成",
        "failed": "需补充/中断",
    }

    lines = ["开始 -> 节点迭代 -> 结束", ""]
    for key in seen_keys:
        event = latest_by_key.get(key, {})
        status = event.get("status", "pending")
        icon = status_icon.get(status, "⚪")
        label = status_text.get(status, "待执行")
        title = event.get("title") or key
        detail = (event.get("detail") or "").strip()
        if detail:
            lines.append(f"- {icon} **{title}** ｜ {label} ｜ {detail}")
        else:
            lines.append(f"- {icon} **{title}** ｜ {label}")

    return "\n".join(lines)


def build_thinking_markdown(module_mode: str, events: list[dict]) -> str:
    """构建“思考过程”段落：按事件逐行追加。"""
    if not events:
        return "- 等待开始..."

    icon_map = {
        "running": "🟡",
        "done": "🟢",
        "failed": "🔴",
        "pending": "⚪",
    }

    status_text = {
        "running": "进行中",
        "done": "完成",
        "failed": "中断",
        "pending": "待执行",
    }

    lines = []
    for idx, event in enumerate(events, start=1):
        status = event.get("status", "pending")
        icon = icon_map.get(status, "⚪")
        title = event.get("title") or event.get("key") or "未命名节点"
        detail = (event.get("detail") or "").strip()
        if detail:
            lines.append(f"{idx}. {icon} {title}（{status_text.get(status, '待执行')}）: {detail}")
        else:
            lines.append(f"{idx}. {icon} {title}（{status_text.get(status, '待执行')}）")

    return "\n".join(lines)


def build_structured_response(module_mode: str, events: list[dict], final_output: str) -> str:
    """把思考过程、工作流、最终输出拼成可保存的单条消息。"""
    thinking_md = build_thinking_markdown(module_mode, events)
    workflow_md = build_workflow_markdown(module_mode, events)
    safe_final = final_output or "（正在生成中...）"

    return (
        "=========思考过程==========\n\n"
        "<div style='color:#4b5563'>\n\n"
        f"{thinking_md}\n\n"
        "</div>\n\n"
        "=========工作流============\n\n"
        f"{workflow_md}\n\n"
        "==========最终输出===========\n\n"
        f"{safe_final}"
    )


with st.sidebar:
    # 侧边栏负责会话管理，体验上接近常见大模型产品的历史会话区。
    st.markdown("## 模块切换")
    st.session_state["module_mode"] = st.radio(
        "选择模式",
        options=["医疗分诊对话模块", "心理咨询模块"],
        index=0 if st.session_state.get("module_mode") == "医疗分诊对话模块" else 1,
    )

    st.markdown("## 会话管理")
    sidebar_action_cols = st.columns(2)
    if sidebar_action_cols[0].button("新建会话", use_container_width=True):
        create_new_chat()
        st.rerun()
    if sidebar_action_cols[1].button("删除当前", use_container_width=True):
        delete_current_chat()
        st.rerun()

    st.caption("历史会话")
    current_session = get_current_session()

    st.markdown("### 输出风格")
    st.session_state["tone_mode"] = st.selectbox(
        "语气",
        options=["温和", "理性"],
        index=0 if st.session_state["tone_mode"] == "温和" else 1,
    )
    st.session_state["output_mode"] = st.selectbox(
        "模式",
        options=["简短模式", "思考模式"],
        index=0 if st.session_state["output_mode"] == "简短模式" else 1,
    )

    for session in st.session_state["sessions"]:
        label = session["title"] or "新对话"
        if st.button(
            label,
            key=f"session_{session['id']}",
            use_container_width=True,
            type="primary" if session["id"] == current_session["id"] else "secondary",
        ):
            switch_session(session["id"])
            st.rerun()

    st.markdown("### 文件记忆上传")
    uploaded = st.file_uploader("上传补充材料（txt）", type=["txt"], accept_multiple_files=False)
    if uploaded is not None:
        raw = uploaded.read()
        upload_sig = f"{current_session['id']}::{uploaded.name}::{len(raw)}"
        if upload_sig != st.session_state.get("last_uploaded_sig", ""):
            summary = summarize_uploaded_text(uploaded.name, raw)
            add_upload_memory(
                session_id=current_session["id"],
                filename=uploaded.name,
                summary=summary,
                saved_path="",
            )
            st.session_state["last_uploaded_sig"] = upload_sig
            st.success("上传内容已写入会话记忆。")

    memory_items = get_upload_memory(current_session["id"])
    if memory_items:
        with st.expander("查看会话记忆中的上传文件", expanded=False):
            for item in memory_items[-5:]:
                st.write(f"- {item.get('filename', '未知文件')}：{item.get('summary', '')[:120]}")

    st.markdown("### 管理员知识库管理")
    admin_token_input = st.text_input("管理员令牌", type="password")
    is_admin = bool(admin_token_input and admin_token_input == os.getenv("ADMIN_TOKEN", ""))
    if is_admin:
        kb_files = []
        for root, _, filenames in os.walk(get_abs_path("data")):
            for name in filenames:
                if name.lower().endswith((".txt", ".pdf")):
                    kb_files.append(os.path.relpath(os.path.join(root, name), get_abs_path("data")))
        kb_files = sorted(kb_files)
        st.caption(f"当前知识文件数：{len(kb_files)}")
        if kb_files:
            selected_file = st.selectbox("删除知识文件", options=kb_files)
            if st.button("删除选中文件", use_container_width=True):
                os.remove(get_abs_path(f"data/{selected_file}"))
                st.success("已删除文件。")
                st.rerun()

        admin_upload = st.file_uploader("上传知识库文件（txt/pdf）", type=["txt", "pdf"], key="admin_kb_upload")
        if admin_upload is not None and st.button("保存知识库文件", use_container_width=True):
            save_path = get_abs_path(f"data/{admin_upload.name}")
            with open(save_path, "wb") as f:
                f.write(admin_upload.read())
            st.success("知识文件上传成功。")
            st.rerun()

        if st.button("管理员重建知识库", use_container_width=True):
            try:
                rag_service.vector_store.reset_store(clear_md5=True)
                rag_service.vector_store.load_document(force_reload=True)
                rag_service._collection_ready_checked = True
                st.success("知识库重建完成。")
            except Exception as e:
                logger.error(f"管理员重建知识库失败: {str(e)}", exc_info=True)
                st.error("知识库重建失败。")

    if st.session_state.get("module_mode") == "心理咨询模块":
        st.markdown("### 云端OCR配置说明")
        st.caption("推荐使用讯飞PDF OCR：请在 config/psych.yaml 填写 xfyun_app_id、xfyun_api_secret、xfyun_base_url。")
        st.caption("也可通过环境变量覆盖：XFYUN_OCR_APP_ID、XFYUN_OCR_API_SECRET。")

st.markdown(
    f"""
    <div class="hero-wrap">
        <h1 class="hero-title">{'医疗分诊对话助手' if st.session_state.get('module_mode') == '医疗分诊对话模块' else '心理咨询模块（GraphRAG）'}</h1>
        <p class="hero-sub">{'围绕症状收集、严重程度判断与分级建议，支持多轮补充与记忆整合。' if st.session_state.get('module_mode') == '医疗分诊对话模块' else '基于《伯恩斯新情绪疗法》构建专用 GraphRAG，先全局检索再局部检索，聚焦心理困扰支持。'}</p>
        <span class="stat">{'症状采集' if st.session_state.get('module_mode') == '医疗分诊对话模块' else '模糊度判断'}</span>
        <span class="stat">{'风险分级' if st.session_state.get('module_mode') == '医疗分诊对话模块' else 'Global Search'}</span>
        <span class="stat">{'就医建议' if st.session_state.get('module_mode') == '医疗分诊对话模块' else 'Local Search'}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
action_cols = st.columns([1, 1, 4])
if action_cols[0].button("清空会话"):
    # 清空的是“当前会话”的消息，不影响其他历史会话。
    persist_current_messages([])
    st.session_state["pending_prompt"] = ""
    st.rerun()

if action_cols[1].button("重建知识库"):
    try:
        with st.spinner("正在重建知识库，请稍候..."):
            if st.session_state.get("module_mode") == "医疗分诊对话模块":
                # 这里直接调用当前运行中的 RAG 服务实例，避免页面重启后才生效。
                rag_service.vector_store.reset_store(clear_md5=True)
                rag_service.vector_store.load_document(force_reload=True)
                rag_service._collection_ready_checked = True
            else:
                st.session_state["psych_rag_service"].rebuild()
        st.success("知识库重建完成。")
    except Exception as e:
        logger.error(f"知识库重建失败: {str(e)}", exc_info=True)
        st.error("知识库重建失败，请查看日志。")

shortcut_cols = st.columns(3)
shortcuts = (
    [
        "我头疼，已经持续一天了",
        "发烧38.5度并伴随肌肉酸痛",
        "头痛欲裂还伴有呕吐，怎么办？",
    ]
    if st.session_state.get("module_mode") == "医疗分诊对话模块"
    else [
        "我总觉得自己做什么都不够好，越想越难受",
        "我会反复自责，一件小事也会脑补成最坏结果",
        "我最近总是失眠，还一直担心明天会更糟",
    ]
)
for col, text in zip(shortcut_cols, shortcuts):
    if col.button(text):
        st.session_state["pending_prompt"] = text

current_session = get_current_session()
current_messages = current_session.get("messages", [])

# 页面展示的始终是“当前会话”的消息。
if not current_messages:
    st.info("可以先试试上面的快捷问题，也可以直接在下方输入你的需求。")

for message in current_messages:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        render_message(message)

input_prompt = st.chat_input(
    "请输入你的症状描述，例如：轻微头痛24小时，无呕吐发烧"
    if st.session_state.get("module_mode") == "医疗分诊对话模块"
    else "请输入你的心理困扰，例如：最近两周总是紧张，影响睡眠和工作"
)
prompt = input_prompt or st.session_state.get("pending_prompt", "")
if prompt:
    st.session_state["pending_prompt"] = ""
    with st.chat_message("user", avatar="🧑"):
        st.write(prompt)
    # 先写入用户消息，再调用 Agent，这样异常时也能保留用户输入。
    current_messages = current_messages + [{"role": "user", "content": prompt}]
    persist_current_messages(current_messages)

    response_chunks = []

    try:
        with st.spinner("正在分析问题并检索答案..."):
            tone = "mild" if st.session_state.get("tone_mode") == "温和" else "rational"
            mode = "short" if st.session_state.get("output_mode") == "简短模式" else "think"
            trace_events = []
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()

                def refresh_render():
                    merged = build_structured_response(
                        st.session_state.get("module_mode"),
                        trace_events,
                        "".join(response_chunks),
                    )
                    body, _ = split_response_and_references(merged)
                    response_placeholder.markdown(body or merged, unsafe_allow_html=True)

                trace_refresh_interval = float(agent_conf.get("trace_refresh_interval_sec", 0.0))
                last_trace_render = [0.0]

                def on_trace(event: dict):
                    trace_events.append(event)
                    now = time.time()
                    if trace_refresh_interval <= 0 or (now - last_trace_render[0]) >= trace_refresh_interval:
                        refresh_render()
                        last_trace_render[0] = now

                if st.session_state.get("module_mode") == "医疗分诊对话模块":
                    res_stream = st.session_state["agent"].execute_stream(
                        current_messages,
                        session_id=current_session["id"],
                        tone=tone,
                        mode=mode,
                        trace_callback=on_trace,
                    )
                else:
                    res_stream = st.session_state["psych_agent"].execute_stream(
                        current_messages,
                        trace_callback=on_trace,
                    )

                refresh_render()
                for chunk in res_stream:
                    response_chunks.append(chunk)
                    refresh_render()

        response_text = "".join(response_chunks).strip()
        if not response_text:
            response_text = "暂时没有生成有效回答，请重试。"

        structured_response = build_structured_response(
            st.session_state.get("module_mode"),
            trace_events,
            response_text,
        )
        body, references = split_response_and_references(structured_response)
        response_placeholder.markdown(body or structured_response, unsafe_allow_html=True)
        render_references(references)
    except Exception as e:
        logger.error(f"对话处理失败: {str(e)}", exc_info=True)
        response_text = "服务暂时不可用，请稍后重试。"
        structured_response = build_structured_response(
            st.session_state.get("module_mode"),
            [],
            response_text,
        )
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(structured_response, unsafe_allow_html=True)

    # 最终回答也要落盘，这样刷新页面后仍能恢复完整会话。
    current_messages = current_messages + [{"role": "assistant", "content": structured_response}]
    persist_current_messages(current_messages)
    st.rerun()
