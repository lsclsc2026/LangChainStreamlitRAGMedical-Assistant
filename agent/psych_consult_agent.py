import json
import re
from typing import Callable

from model.factory import get_chat_model
from psych.graphrag_service import PsychGraphRagService
from utils.prompt_loader import load_psych_answer_prompt, load_psych_clarity_judge_prompt


class PsychConsultAgent:
    """心理咨询模块：模糊判断 -> Global Search -> 引导补充 -> Local Search -> 输出。"""

    def __init__(self):
        self.model = get_chat_model()
        self.graph_rag = PsychGraphRagService()
        self.judge_prompt = load_psych_clarity_judge_prompt()
        self.answer_prompt = load_psych_answer_prompt()

    @staticmethod
    def _aggregate_user_input(messages: list[dict]) -> str:
        blocks = []
        for msg in messages:
            if msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if content:
                    blocks.append(content)
        return "\n\n".join(blocks)

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    @staticmethod
    def _count_user_turns(messages: list[dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "user" and (m.get("content") or "").strip())

    @staticmethod
    def _info_points(text: str) -> int:
        content = (text or "").lower()
        rules = [
            r"焦虑|紧张|害怕|恐惧|惊恐|担心",
            r"持续\s*\d+\s*(天|周|月|小时)|\d+\s*(天|周|月|小时)",
            r"在.*时候|场景|触发|一到",
            r"心慌|心悸|胸闷|出汗|发抖|失眠|头晕",
            r"影响.*(工作|学习|社交|睡眠|生活)",
            r"没有.*(自杀|自伤)|有.*(自杀|自伤)|绝望",
        ]
        return sum(1 for p in rules if re.search(p, content))

    def _llm_text(self, template: str, **kwargs) -> str:
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace("{" + key + "}", str(value))

        response = self.model.invoke(rendered)
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            return "\n".join(text_parts).strip()
        return str(content).strip()

    def _llm_stream(self, template: str, **kwargs):
        rendered = template
        for key, value in kwargs.items():
            rendered = rendered.replace("{" + key + "}", str(value))

        for chunk in self.model.stream(rendered):
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                if content:
                    yield content
                continue

            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                merged = "".join(text_parts)
                if merged:
                    yield merged

    @staticmethod
    def _docs_to_context(docs) -> str:
        blocks = []
        for idx, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "未知来源")
            page = d.metadata.get("page")
            blocks.append(f"[G{idx}] 来源={src} 页码={page}\n{d.page_content.strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def _local_to_context(nodes: list[dict]) -> str:
        blocks = []
        for idx, n in enumerate(nodes, start=1):
            blocks.append(
                f"[L{idx}] node={n.get('node_id')} score={n.get('score')} source={n.get('source')} page={n.get('page')}\n{n.get('text', '').strip()}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _trace(trace_callback: Callable[[dict], None] | None, key: str, title: str, status: str, detail: str = ""):
        if trace_callback is None:
            return
        trace_callback(
            {
                "key": key,
                "title": title,
                "status": status,
                "detail": detail,
            }
        )

    def execute_stream(self, messages: list[dict], trace_callback: Callable[[dict], None] | None = None):
        self._trace(trace_callback, "collect", "汇总心理困扰描述", "running", "正在汇总用户多轮输入")
        user_input = self._aggregate_user_input(messages)
        if not user_input:
            self._trace(trace_callback, "collect", "汇总心理困扰描述", "failed", "当前没有可用输入")
            yield "可以先告诉我：你最近最困扰的情绪是什么？通常在什么场景下出现？\n"
            return
        self._trace(trace_callback, "collect", "汇总心理困扰描述", "done", "已获取可分析文本")

        self._trace(trace_callback, "global", "Global Search 全局检索", "running", "从心理知识图谱召回候选内容")
        global_docs = self.graph_rag.global_search(user_input, k=8)
        global_context = self._docs_to_context(global_docs)
        self._trace(trace_callback, "global", "Global Search 全局检索", "done", f"候选片段={len(global_docs)}")

        self._trace(trace_callback, "clarity", "模糊度判断与追问决策", "running", "评估是否可进入局部检索")
        judge_text = self._llm_text(
            self.judge_prompt + "\n\n用户累计输入:\n{user_input}\n\n全局检索摘要:\n{global_context}",
            user_input=user_input,
            global_context=global_context[:6000],
        )
        judge = self._extract_json(judge_text)

        user_turns = self._count_user_turns(messages)
        info_points = self._info_points(user_input)
        force_local = user_turns >= 3 or info_points >= 3

        ready = bool(judge.get("is_ready_for_local", False) or force_local)
        if not ready:
            self._trace(
                trace_callback,
                "clarity",
                "模糊度判断与追问决策",
                "failed",
                f"信息不足：轮次={user_turns}，信息点={info_points}",
            )
            follow_up = (judge.get("follow_up_question") or "").strip()
            if not follow_up:
                follow_up = "我想更准确地帮助你：这种焦虑/恐惧通常在什么情境出现？大概持续多久，是否影响睡眠或工作？"
            yield follow_up + "\n"
            return
        self._trace(
            trace_callback,
            "clarity",
            "模糊度判断与追问决策",
            "done",
            f"准入局部检索：轮次={user_turns}，信息点={info_points}",
        )

        self._trace(trace_callback, "local", "Local Search 局部检索", "running", "聚焦高相关节点并构建证据")
        local_nodes = self.graph_rag.local_search(user_input, global_docs, max_nodes=8)
        local_context = self._local_to_context(local_nodes)
        self._trace(trace_callback, "local", "Local Search 局部检索", "done", f"命中节点={len(local_nodes)}")

        self._trace(trace_callback, "answer", "生成心理支持答复", "running", "结合Global/Local上下文生成建议")
        has_output = False
        for chunk in self._llm_stream(
            self.answer_prompt
            + "\n\n用户累计输入:\n{user_input}\n\nGlobal Context:\n{global_context}\n\nLocal Context:\n{local_context}",
            user_input=user_input,
            global_context=global_context[:6000],
            local_context=local_context[:7000],
        ):
            has_output = True
            yield chunk

        if not has_output:
            yield "暂时没有生成有效回答，请稍后重试。\n"
        self._trace(trace_callback, "answer", "生成心理支持答复", "done", "流程结束")
