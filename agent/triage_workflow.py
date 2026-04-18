import json
import re
from typing import Callable, Iterable

from agent.tools.agent_tools import open_hospital_map
from model.factory import get_chat_model
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf
from utils.prompt_loader import (
    load_final_tone_prompt,
    load_kb_base_prompt,
    load_route_output_prompt,
    load_severity_judge_prompt,
    load_sufficiency_judge_prompt,
)


class TriageWorkflow:
    """分诊编排器：信息充足度判断 -> 严重程度分级 -> 路由建议输出。"""

    def __init__(self):
        self.model = get_chat_model()
        self.rag = RagSummarizeService()
        self.kb_base_prompt = load_kb_base_prompt()
        self.sufficiency_prompt = load_sufficiency_judge_prompt()
        self.severity_prompt = load_severity_judge_prompt()
        self.route_prompt = load_route_output_prompt()
        self.final_tone_prompt = load_final_tone_prompt()
        perf_conf = agent_conf.get("triage_perf", {})
        self.enable_lightweight_sufficiency = bool(perf_conf.get("enable_lightweight_sufficiency", True))
        self.enable_lightweight_severity = bool(perf_conf.get("enable_lightweight_severity", True))
        self.enable_single_pass_generation = bool(perf_conf.get("enable_single_pass_generation", True))
        self.route_top_k = int(perf_conf.get("route_top_k", 1))

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = (text or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _aggregate_user_inputs(messages: Iterable[dict], upload_memory: list[dict] | None = None) -> str:
        user_blocks = []
        for msg in messages:
            if msg.get("role") == "user":
                content = (msg.get("content") or "").strip()
                if content:
                    user_blocks.append(content)

        for item in upload_memory or []:
            filename = (item.get("filename") or "").strip()
            summary = (item.get("summary") or "").strip()
            if filename or summary:
                user_blocks.append(f"[上传文件] {filename}\n{summary}")

        return "\n\n".join(user_blocks)

    @staticmethod
    def _extract_dimensions(text: str) -> set[str]:
        lower = (text or "").lower()
        dims = set()
        if re.search(r"体温|发热|发烧|测量", lower):
            dims.add("fever")
        if re.search(r"呕吐|恶心", lower):
            dims.add("vomiting")
        if re.search(r"畏光|怕光", lower):
            dims.add("photophobia")
        if re.search(r"视力模糊|视物模糊|复视|视力", lower):
            dims.add("vision")
        if re.search(r"持续|多久|天|小时|分钟", lower):
            dims.add("duration")
        if re.search(r"轻微|中度|严重|剧烈|头痛欲裂|疼痛程度", lower):
            dims.add("pain_severity")
        if re.search(r"胀痛|刺痛|钝痛|跳痛|压痛", lower):
            dims.add("pain_type")
        if re.search(r"肌肉酸痛|酸痛", lower):
            dims.add("myalgia")
        return dims

    @staticmethod
    def _is_negative_short_reply(text: str) -> bool:
        reply = (text or "").strip()
        if not reply:
            return False
        compact = re.sub(r"[。！!？，,\s]", "", reply)
        return compact in {"没有", "无", "不是", "未", "没"}

    @staticmethod
    def _build_fact_lines_from_qa(messages: list[dict]) -> list[str]:
        """将“助手追问 + 用户短答”补全为结构化事实，减少重复追问。"""
        fact_lines: list[str] = []
        dimension_facts: dict[str, str] = {}

        for i in range(1, len(messages)):
            prev = messages[i - 1]
            curr = messages[i]
            if prev.get("role") != "assistant" or curr.get("role") != "user":
                continue

            question = (prev.get("content") or "").strip()
            answer = (curr.get("content") or "").strip()
            if not question or not answer:
                continue

            asked_dims = TriageWorkflow._extract_dimensions(question)
            if not asked_dims:
                continue

            lower_answer = answer.lower()
            negative_short = TriageWorkflow._is_negative_short_reply(answer)

            if "fever" in asked_dims:
                if re.search(r"没量|未量|没测|未测", lower_answer):
                    dimension_facts["fever"] = "用户未测量体温"
                elif negative_short or re.search(r"不发热|无发热|没有发热|没有发烧", lower_answer):
                    dimension_facts["fever"] = "用户无发热"
                elif re.search(r"发热|发烧|体温", lower_answer):
                    dimension_facts["fever"] = f"用户体温信息：{answer}"

            for dim, negative_text, positive_pat, positive_prefix in [
                ("vomiting", "用户无呕吐", r"呕吐|恶心", "用户存在呕吐/恶心："),
                ("photophobia", "用户无畏光", r"畏光|怕光", "用户存在畏光："),
                ("vision", "用户无视力模糊", r"视力|视物模糊|复视", "用户存在视觉相关症状："),
            ]:
                if dim not in asked_dims:
                    continue
                if negative_short or re.search(r"无|没有|未见|否认", lower_answer):
                    dimension_facts[dim] = negative_text
                elif re.search(positive_pat, lower_answer):
                    dimension_facts[dim] = positive_prefix + answer

            if "duration" in asked_dims:
                if re.search(r"\d+\s*(分钟|小时|天|周|月)|持续", lower_answer):
                    dimension_facts["duration"] = f"用户症状时长：{answer}"

            if "pain_severity" in asked_dims:
                if re.search(r"轻微|中度|严重|剧烈|难以忍受", lower_answer):
                    dimension_facts["pain_severity"] = f"用户症状程度：{answer}"

        for value in dimension_facts.values():
            fact_lines.append(value)
        return fact_lines

    @classmethod
    def _build_enhanced_user_input(cls, messages: list[dict], upload_memory: list[dict] | None = None) -> str:
        merged = cls._aggregate_user_inputs(messages, upload_memory)
        fact_lines = cls._build_fact_lines_from_qa(messages)
        if not fact_lines:
            return merged
        return merged + "\n\n[问答补全事实]\n" + "\n".join(f"- {line}" for line in fact_lines)

    @classmethod
    def _refine_follow_up_question(cls, question: str, merged_user_input: str) -> str:
        """避免追问已知维度；若重复则改写为未知维度组合提问。"""
        q = (question or "").strip()
        if not q:
            return q

        known_slots = cls._collect_information_slots(merged_user_input)
        asked_dims = cls._extract_dimensions(q)
        if not asked_dims:
            return q

        all_known = all(known_slots.get(dim, False) for dim in asked_dims if dim in known_slots)
        if not all_known:
            return q

        options = []
        if not known_slots.get("duration"):
            options.append("症状大概持续了多久")
        if not known_slots.get("pain_severity"):
            options.append("目前头痛属于轻微、中度还是剧烈")
        if not known_slots.get("fever"):
            options.append("是否测量过体温，若未测可直接告诉我“未测”")
        if not known_slots.get("vomiting"):
            options.append("是否伴有呕吐或恶心")
        if not known_slots.get("photophobia"):
            options.append("是否有畏光")
        if not known_slots.get("vision"):
            options.append("是否有视力模糊或复视")
        if not known_slots.get("myalgia"):
            options.append("是否伴随肌肉酸痛")

        if not options:
            return "目前信息已基本够用，我将先进行风险评估；若你愿意，也可补充是否影响睡眠或工作。"

        return "为避免遗漏，请再补充：" + "；".join(options[:2]) + "。"

    def _llm_text(self, template: str, **kwargs) -> str:
        # 业务提示词里包含 JSON 花括号示例，不能交给 PromptTemplate 解析。
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
        """模型原生流式输出。"""
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

    def judge_sufficiency(self, merged_user_input: str) -> dict:
        rules_context = self.rag.query_knowledge(merged_user_input, "规则知识库", top_k=2)
        content = self._llm_text(
            self.sufficiency_prompt + "\n\n用户累计输入:\n{merged_user_input}\n\n规则知识库片段:\n{rules_context}",
            merged_user_input=merged_user_input,
            rules_context=rules_context or "未检索到规则知识库片段",
        )
        parsed = self._extract_json(content)
        return {
            "is_sufficient": bool(parsed.get("is_sufficient", False)),
            "missing_fields": parsed.get("missing_fields", []),
            "follow_up_question": (parsed.get("follow_up_question") or "").strip(),
            "reason": (parsed.get("reason") or "").strip(),
            "rules_context": rules_context,
        }

    def judge_severity(self, merged_user_input: str) -> dict:
        severity_context = self.rag.query_knowledge(merged_user_input, "严重程度知识库", top_k=2)
        content = self._llm_text(
            self.severity_prompt + "\n\n用户累计输入:\n{merged_user_input}\n\n严重程度知识库片段:\n{severity_context}",
            merged_user_input=merged_user_input,
            severity_context=severity_context or "未检索到严重程度知识库片段",
        )
        parsed = self._extract_json(content)
        severity = (parsed.get("severity") or "mild").strip().lower()
        if severity not in {"mild", "serious", "critical"}:
            severity = "mild"
        return {
            "severity": severity,
            "reason": (parsed.get("reason") or "").strip(),
            "risk_flags": parsed.get("risk_flags", []),
            "severity_context": severity_context,
        }

    @staticmethod
    def _route_knowledges(severity: str) -> list[str]:
        if severity == "critical":
            return ["就诊建议知识库"]
        if severity == "serious":
            return ["生活习惯知识库", "饮食知识库", "中成药推荐知识库"]
        return ["生活习惯知识库", "情绪管理知识库", "饮食知识库"]

    def _format_route_context(self, contexts: dict[str, str]) -> str:
        blocks = []
        for kb_name, kb_text in contexts.items():
            blocks.append(f"[{kb_name}]\n{kb_text or '未检索到该知识库内容'}")
        return "\n\n".join(blocks)

    def _apply_final_tone(self, routed_answer: str, tone: str, mode: str) -> str:
        return self._llm_text(
            self.final_tone_prompt + "\n\n原始路由建议:\n{raw_answer}\n\n控制参数:\ntone={tone}\nmode={mode}",
            raw_answer=routed_answer,
            tone=tone,
            mode=mode,
        )

    @staticmethod
    def _missing_fields_from_slots(slots: dict[str, bool]) -> list[str]:
        order = [
            ("pain_severity", "症状程度"),
            ("duration", "持续时间"),
            ("fever", "体温/发热信息"),
            ("vomiting", "呕吐/恶心"),
            ("photophobia", "畏光"),
            ("vision", "视力相关症状"),
            ("myalgia", "肌肉酸痛"),
        ]
        return [label for key, label in order if not slots.get(key)]

    def _quick_sufficiency(self, merged_user_input: str, user_turns: int, info_points: int) -> dict:
        slots = self._collect_information_slots(merged_user_input)
        missing_fields = self._missing_fields_from_slots(slots)
        is_sufficient = user_turns >= 3 or info_points >= 3
        follow_up_question = ""
        if not is_sufficient:
            asks = []
            if "持续时间" in missing_fields:
                asks.append("症状大概持续了多久")
            if "症状程度" in missing_fields:
                asks.append("目前不适程度是轻微、中度还是剧烈")
            if "体温/发热信息" in missing_fields:
                asks.append("是否测量过体温，若未测可直接回复“未测”")
            if "呕吐/恶心" in missing_fields:
                asks.append("是否伴有呕吐或恶心")
            if "畏光" in missing_fields:
                asks.append("是否有畏光")
            if "视力相关症状" in missing_fields:
                asks.append("是否有视力模糊或复视")
            follow_up_question = "为更准确判断，请补充：" + "；".join(asks[:2]) + "。"

        return {
            "is_sufficient": is_sufficient,
            "missing_fields": missing_fields,
            "follow_up_question": follow_up_question,
            "reason": f"规则快判：轮次={user_turns}，信息点={info_points}",
            "rules_context": "",
        }

    @staticmethod
    def _quick_severity(merged_user_input: str) -> dict:
        text = (merged_user_input or "").lower()
        critical_patterns = [
            r"头痛欲裂|最严重头痛|爆发性头痛",
            r"意识异常|昏厥|抽搐|言语不清|肢体无力",
            r"呼吸困难|胸痛",
            r"喷射性呕吐",
            r"持续视力下降|复视",
        ]
        serious_patterns = [
            r"发热|发烧|体温|38|39|40",
            r"呕吐|恶心",
            r"肌肉酸痛|酸痛",
            r"影响睡眠|影响工作|影响日常",
        ]

        if any(re.search(p, text) for p in critical_patterns):
            return {
                "severity": "critical",
                "reason": "规则快判命中危急信号",
                "risk_flags": ["危急信号"],
                "severity_context": "",
            }

        serious_hits = sum(1 for p in serious_patterns if re.search(p, text))
        if serious_hits >= 2:
            return {
                "severity": "serious",
                "reason": f"规则快判命中中高风险特征={serious_hits}",
                "risk_flags": ["中高风险特征"],
                "severity_context": "",
            }

        return {
            "severity": "mild",
            "reason": "规则快判未命中高风险特征",
            "risk_flags": [],
            "severity_context": "",
        }

    def _build_single_pass_prompt(self) -> str:
        return (
            self.kb_base_prompt
            + "\n\n"
            + self.route_prompt
            + "\n\n"
            + "请直接输出最终回答，不要展示内部分析过程。"
            + "\n输出要求：结论优先，随后给出可执行建议、观察指标和就医触发条件。"
            + "\n控制参数：tone={tone}，mode={mode}。"
            + "\n\n用户累计输入:\n{merged_user_input}\n\n分级结果:\n{severity_json}\n\n路由知识片段:\n{route_context}"
        )

    @staticmethod
    def _count_user_turns(messages: list[dict]) -> int:
        return sum(1 for msg in messages if msg.get("role") == "user" and (msg.get("content") or "").strip())

    @staticmethod
    def _collect_information_slots(merged_user_input: str) -> dict[str, bool]:
        text = (merged_user_input or "").lower()
        slots = {
            "pain_severity": bool(re.search(r"轻微|中度|严重|剧烈|难以忍受|头痛欲裂", text)),
            "vomiting": bool(re.search(r"呕吐|恶心|无呕吐|没有呕吐|未呕吐", text)),
            "fever": bool(re.search(r"发烧|发热|体温|38|39|40|未量体温|没量体温", text)),
            "myalgia": bool(re.search(r"肌肉酸痛|酸痛", text)),
            "duration": bool(re.search(r"持续\s*\d+\s*(分钟|小时|天)|\d+\s*(分钟|小时|天)|多久", text)),
            "pain_type": bool(re.search(r"胀痛|刺痛|钝痛|跳痛|压痛", text)),
            "vision": bool(re.search(r"视力模糊|无视力模糊|没有视力模糊", text)),
            "photophobia": bool(re.search(r"怕光|畏光|无怕光|不怕光|没有怕光", text)),
        }
        return slots

    @classmethod
    def _count_information_points(cls, merged_user_input: str) -> int:
        slots = cls._collect_information_slots(merged_user_input)
        return sum(1 for value in slots.values() if value)

    @staticmethod
    def _trace(trace_callback: Callable[[dict], None] | None, key: str, title: str, status: str, detail: str = "") -> None:
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

    def run(
        self,
        messages: list[dict],
        upload_memory: list[dict] | None = None,
        tone: str = "mild",
        mode: str = "think",
        city: str = "",
        trace_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        self._trace(trace_callback, "collect", "收集用户信息", "running", "正在汇总历史对话与上传记忆")
        merged_user_input = self._build_enhanced_user_input(messages, upload_memory)
        if not merged_user_input:
            self._trace(trace_callback, "collect", "收集用户信息", "failed", "当前没有可用输入")
            return {
                "status": "insufficient",
                "message": "请先描述你的主要症状、持续时长和已尝试处理方式。",
                "diagnostics": {},
            }
        self._trace(trace_callback, "collect", "收集用户信息", "done", "已汇总多轮信息")

        user_turns = self._count_user_turns(messages)
        info_points = self._count_information_points(merged_user_input)
        force_progress = user_turns >= 3 or info_points >= 3

        self._trace(
            trace_callback,
            "sufficiency",
            "信息充足度判断",
            "running",
            f"用户轮次={user_turns}，信息点={info_points}",
        )
        if force_progress:
            sufficiency = {
                "is_sufficient": True,
                "missing_fields": [],
                "follow_up_question": "",
                "reason": f"触发硬阈值推进：用户轮次={user_turns}，信息点={info_points}",
                "rules_context": "",
            }
            self._trace(trace_callback, "sufficiency", "信息充足度判断", "done", sufficiency["reason"])
        else:
            sufficiency = (
                self._quick_sufficiency(merged_user_input, user_turns, info_points)
                if self.enable_lightweight_sufficiency
                else self.judge_sufficiency(merged_user_input)
            )
            if not sufficiency["is_sufficient"]:
                self._trace(
                    trace_callback,
                    "sufficiency",
                    "信息充足度判断",
                    "failed",
                    sufficiency.get("reason") or "信息不足，需要追问",
                )
                follow_up = sufficiency["follow_up_question"] or "目前信息还不足，请补充：主要不适表现、持续多久、是否影响日常生活？"
                follow_up = self._refine_follow_up_question(follow_up, merged_user_input)
                return {
                    "status": "insufficient",
                    "message": follow_up,
                    "diagnostics": {
                        "sufficiency": sufficiency,
                        "user_turns": user_turns,
                        "info_points": info_points,
                    },
                }
            self._trace(trace_callback, "sufficiency", "信息充足度判断", "done", sufficiency.get("reason") or "信息已充足")

        self._trace(trace_callback, "severity", "严重程度评估", "running", "结合风险词与知识库进行分级")
        severity = self._quick_severity(merged_user_input) if self.enable_lightweight_severity else self.judge_severity(merged_user_input)
        self._trace(
            trace_callback,
            "severity",
            "严重程度评估",
            "done",
            f"分级结果={severity.get('severity', 'mild')}",
        )

        self._trace(trace_callback, "route", "路由知识检索", "running", "按分级选择知识库并召回证据")
        route_kbs = self._route_knowledges(severity["severity"])
        route_contexts = self.rag.query_multiple_knowledges(merged_user_input, route_kbs, top_k=self.route_top_k)
        self._trace(trace_callback, "route", "路由知识检索", "done", f"命中知识库: {', '.join(route_kbs)}")

        self._trace(trace_callback, "compose", "生成最终建议", "running", "基于路由证据组织回答")
        if self.enable_single_pass_generation:
            final_answer = self._llm_text(
                self._build_single_pass_prompt(),
                merged_user_input=merged_user_input,
                severity_json=json.dumps(severity, ensure_ascii=False),
                route_context=self._format_route_context(route_contexts),
                tone=tone,
                mode=mode,
            )
            self._trace(trace_callback, "compose", "生成最终建议", "done", "单次生成完成")
        else:
            routed_answer = self._llm_text(
                self.kb_base_prompt
                + "\n\n"
                + self.route_prompt
                + "\n\n用户累计输入:\n{merged_user_input}\n\n分级结果:\n{severity_json}\n\n路由知识片段:\n{route_context}",
                merged_user_input=merged_user_input,
                severity_json=json.dumps(severity, ensure_ascii=False),
                route_context=self._format_route_context(route_contexts),
            )
            self._trace(trace_callback, "compose", "生成最终建议", "done", "已生成首版建议")
            self._trace(trace_callback, "tone", "语气与风格调整", "running", f"tone={tone}, mode={mode}")
            final_answer = self._apply_final_tone(routed_answer, tone=tone, mode=mode)
            self._trace(trace_callback, "tone", "语气与风格调整", "done", "输出风格调整完成")

        if severity["severity"] == "critical":
            self._trace(trace_callback, "tool_map", "工具调用：医院地图", "running", "检索附近就医资源")
            map_links = open_hospital_map.invoke({"city": city or "当前位置", "symptom": "紧急症状"})
            final_answer = (
                f"{final_answer}\n\n"
                f"建议尽快线下就医。可参考以下地图检索医院：\n{map_links}"
            )
            self._trace(trace_callback, "tool_map", "工具调用：医院地图", "done", "地图链接已附加")

        self._trace(trace_callback, "finish", "输出最终答复", "done", "流程结束")

        return {
            "status": "ok",
            "message": final_answer,
            "diagnostics": {
                "sufficiency": sufficiency,
                "severity": severity,
                "route_kbs": route_kbs,
                "user_turns": user_turns,
                "info_points": info_points,
            },
        }

    def run_stream(
        self,
        messages: list[dict],
        upload_memory: list[dict] | None = None,
        tone: str = "mild",
        mode: str = "think",
        city: str = "",
        trace_callback: Callable[[dict], None] | None = None,
    ):
        """分诊流式入口：前置判断同步，最终答复原生流式返回。"""
        self._trace(trace_callback, "collect", "收集用户信息", "running", "正在汇总历史对话与上传记忆")
        merged_user_input = self._build_enhanced_user_input(messages, upload_memory)
        if not merged_user_input:
            self._trace(trace_callback, "collect", "收集用户信息", "failed", "当前没有可用输入")
            yield "请先描述你的主要症状、持续时长和已尝试处理方式。\n"
            return
        self._trace(trace_callback, "collect", "收集用户信息", "done", "已汇总多轮信息")

        user_turns = self._count_user_turns(messages)
        info_points = self._count_information_points(merged_user_input)
        force_progress = user_turns >= 3 or info_points >= 3

        self._trace(
            trace_callback,
            "sufficiency",
            "信息充足度判断",
            "running",
            f"用户轮次={user_turns}，信息点={info_points}",
        )
        if force_progress:
            sufficiency = {
                "is_sufficient": True,
                "missing_fields": [],
                "follow_up_question": "",
                "reason": f"触发硬阈值推进：用户轮次={user_turns}，信息点={info_points}",
                "rules_context": "",
            }
            self._trace(trace_callback, "sufficiency", "信息充足度判断", "done", sufficiency["reason"])
        else:
            sufficiency = (
                self._quick_sufficiency(merged_user_input, user_turns, info_points)
                if self.enable_lightweight_sufficiency
                else self.judge_sufficiency(merged_user_input)
            )
            if not sufficiency["is_sufficient"]:
                self._trace(
                    trace_callback,
                    "sufficiency",
                    "信息充足度判断",
                    "failed",
                    sufficiency.get("reason") or "信息不足，需要追问",
                )
                follow_up = sufficiency["follow_up_question"] or "目前信息还不足，请补充：主要不适表现、持续多久、是否影响日常生活？"
                follow_up = self._refine_follow_up_question(follow_up, merged_user_input)
                yield follow_up + "\n"
                return
            self._trace(trace_callback, "sufficiency", "信息充足度判断", "done", sufficiency.get("reason") or "信息已充足")

        self._trace(trace_callback, "severity", "严重程度评估", "running", "结合风险词与知识库进行分级")
        severity = self._quick_severity(merged_user_input) if self.enable_lightweight_severity else self.judge_severity(merged_user_input)
        self._trace(
            trace_callback,
            "severity",
            "严重程度评估",
            "done",
            f"分级结果={severity.get('severity', 'mild')}",
        )

        self._trace(trace_callback, "route", "路由知识检索", "running", "按分级选择知识库并召回证据")
        route_kbs = self._route_knowledges(severity["severity"])
        route_contexts = self.rag.query_multiple_knowledges(merged_user_input, route_kbs, top_k=self.route_top_k)
        self._trace(trace_callback, "route", "路由知识检索", "done", f"命中知识库: {', '.join(route_kbs)}")

        self._trace(trace_callback, "compose", "生成最终建议", "running", "基于路由证据组织回答")

        if self.enable_single_pass_generation:
            for chunk in self._llm_stream(
                self._build_single_pass_prompt(),
                merged_user_input=merged_user_input,
                severity_json=json.dumps(severity, ensure_ascii=False),
                route_context=self._format_route_context(route_contexts),
                tone=tone,
                mode=mode,
            ):
                yield chunk
            self._trace(trace_callback, "compose", "生成最终建议", "done", "单次流式生成完成")
        else:
            routed_answer = self._llm_text(
                self.kb_base_prompt
                + "\n\n"
                + self.route_prompt
                + "\n\n用户累计输入:\n{merged_user_input}\n\n分级结果:\n{severity_json}\n\n路由知识片段:\n{route_context}",
                merged_user_input=merged_user_input,
                severity_json=json.dumps(severity, ensure_ascii=False),
                route_context=self._format_route_context(route_contexts),
            )
            self._trace(trace_callback, "compose", "生成最终建议", "done", "已生成首版建议")

        emergency_appendix = ""
        if severity["severity"] == "critical":
            self._trace(trace_callback, "tool_map", "工具调用：医院地图", "running", "检索附近就医资源")
            map_links = open_hospital_map.invoke({"city": city or "当前位置", "symptom": "紧急症状"})
            emergency_appendix = f"\n\n建议尽快线下就医。可参考以下地图检索医院：\n{map_links}"
            self._trace(trace_callback, "tool_map", "工具调用：医院地图", "done", "地图链接已附加")

        if not self.enable_single_pass_generation:
            self._trace(trace_callback, "tone", "语气与风格调整", "running", f"tone={tone}, mode={mode}")
            for chunk in self._llm_stream(
                self.final_tone_prompt + "\n\n原始路由建议:\n{raw_answer}\n\n控制参数:\ntone={tone}\nmode={mode}",
                raw_answer=routed_answer,
                tone=tone,
                mode=mode,
            ):
                yield chunk
            self._trace(trace_callback, "tone", "语气与风格调整", "done", "输出风格调整完成")

        if emergency_appendix:
            yield emergency_appendix
        self._trace(trace_callback, "finish", "输出最终答复", "done", "流程结束")
