from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def _load_prompt(config_key: str, prompt_name: str) -> str:
    """按配置键读取指定提示词文件。"""
    try:
        prompt_path = get_abs_path(prompts_conf[config_key])
    except KeyError as e:
        logger.error(f"{prompt_name}路径未在配置文件中找到: {str(e)}")
        raise

    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"加载{prompt_name}失败: {str(e)}")
        raise


def load_system_prompts():
    """读取主对话提示词。"""
    return _load_prompt("main_prompt_path", "系统提示词")


def load_rag_prompts():
    """读取 RAG 总结提示词。"""
    return _load_prompt("rag_summarize_prompt_path", "RAG总结提示词")


def load_report_prompts():
    """读取报告生成提示词。"""
    return _load_prompt("report_prompt_path", "报告提示词")


def load_kb_base_prompt():
    """读取知识库拼接基础提示词。"""
    return _load_prompt("kb_base_prompt_path", "知识库基础提示词")


def load_sufficiency_judge_prompt():
    """读取信息充足性判断提示词。"""
    return _load_prompt("sufficiency_judge_prompt_path", "信息充足性判断提示词")


def load_severity_judge_prompt():
    """读取严重程度判断提示词。"""
    return _load_prompt("severity_judge_prompt_path", "严重程度判断提示词")


def load_route_output_prompt():
    """读取路由输出提示词。"""
    return _load_prompt("route_output_prompt_path", "路由输出提示词")


def load_final_tone_prompt():
    """读取最终语气控制提示词。"""
    return _load_prompt("final_tone_prompt_path", "最终语气提示词")


def load_psych_clarity_judge_prompt():
    """读取心理咨询模块的清晰度判断提示词。"""
    return _load_prompt("psych_clarity_judge_prompt_path", "心理清晰度判断提示词")


def load_psych_answer_prompt():
    """读取心理咨询模块的最终回答提示词。"""
    return _load_prompt("psych_answer_prompt_path", "心理咨询输出提示词")

if __name__ == '__main__':
    print(load_rag_prompts())
