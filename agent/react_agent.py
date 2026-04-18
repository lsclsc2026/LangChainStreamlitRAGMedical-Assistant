import re
from typing import Callable, Iterable

from agent.triage_workflow import TriageWorkflow
from utils.upload_memory_store import get_upload_memory



class ReactAgent:
    """Agent 主入口，负责组装模型、工具和会话级上下文。"""
    COMMON_CITIES = [
        "北京", "上海", "广州", "深圳", "杭州", "苏州", "南京", "成都", "重庆", "天津",
        "武汉", "西安", "长沙", "青岛", "宁波", "厦门", "郑州", "合肥", "福州", "济南",
    ]
    INVALID_CITY_VALUES = {"哪个城市", "什么城市", "哪座城市", "哪个市", "哪里", "哪儿"}

    def __init__(self):
        """初始化分诊编排流程。"""
        self.workflow = TriageWorkflow()

    @staticmethod
    def _normalize_messages(messages: Iterable[dict]) -> list[dict]:
        """过滤无效消息，只保留 Agent 能消费的 user/assistant 文本。"""
        normalized = []
        for message in messages:
            role = message.get("role")
            content = (message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    @classmethod
    def _extract_session_facts(cls, messages: list[dict]) -> dict:
        """
        从历史消息里提取稳定事实。

        这里优先抽取“城市”“用户 ID”这类对后续问答影响明显的信息，
        避免模型每次都重新向工具索取，提升多轮对话的一致性。
        """
        facts = {}

        for message in messages:
            content = (message.get("content") or "").strip()
            if not content:
                continue

            for city in cls.COMMON_CITIES:
                if city in content:
                    facts["city"] = city

            city_match = re.search(
                r"(?:在|住在|来自|位于)([^\s，。！？,.!?]{2,12}(?:市|县|区|北京|上海|广州|深圳|杭州|苏州|南京|成都|重庆|天津|武汉|西安|长沙|青岛|宁波|厦门|郑州|合肥|福州|济南))",
                content,
            )
            if city_match:
                candidate_city = city_match.group(1)
                if candidate_city not in cls.INVALID_CITY_VALUES:
                    facts["city"] = candidate_city

            user_id_match = re.search(r"(?:用户ID|ID|id)[：:\s]*([0-9]{3,})", content)
            if user_id_match:
                facts["user_id"] = user_id_match.group(1)

        return facts

    def execute_stream(
        self,
        messages: list[dict],
        session_id: str = "",
        tone: str = "mild",
        mode: str = "think",
        trace_callback: Callable[[dict], None] | None = None,
    ):
        """
        执行一次带历史上下文的流式对话。

        除了原始 messages，还会把抽取出的 session_facts 注入到 runtime context，
        供动态提示词和后续工具调用共同使用。
        """
        normalized_messages = self._normalize_messages(messages)
        if trace_callback:
            trace_callback(
                {
                    "key": "agent_boot",
                    "title": "Agent准备阶段",
                    "status": "running",
                    "detail": "正在整理消息上下文与会话事实",
                }
            )
        session_facts = self._extract_session_facts(normalized_messages)
        upload_memory = get_upload_memory(session_id) if session_id else []
        if trace_callback:
            trace_callback(
                {
                    "key": "agent_boot",
                    "title": "Agent准备阶段",
                    "status": "done",
                    "detail": f"识别城市={session_facts.get('city', '未识别')}，上传记忆={len(upload_memory)}条",
                }
            )

        for chunk in self.workflow.run_stream(
            normalized_messages,
            upload_memory=upload_memory,
            tone=tone,
            mode=mode,
            city=session_facts.get("city", ""),
            trace_callback=trace_callback,
        ):
            yield chunk



if __name__ == '__main__':
    agent = ReactAgent()
    res = agent.execute_stream([{"role": "user", "content": "我头痛24小时，没有呕吐和发烧"}])
    for chunk in res:
        print(chunk, end="", flush=True)
