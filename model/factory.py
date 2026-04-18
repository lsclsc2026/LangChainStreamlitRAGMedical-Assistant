import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Optional, Union
from dotenv import load_dotenv
load_dotenv()


from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from utils.config_handler import rag_conf


def _configure_langsmith() -> None:
    """按配置启用 LangSmith tracing。"""
    if not rag_conf.get("enable_langsmith", False):
        return
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", rag_conf.get("langsmith_project", "langchain-agent-master"))
    os.environ.setdefault("LANGSMITH_ENDPOINT", rag_conf.get("langsmith_endpoint", "https://api.smith.langchain.com"))


_configure_langsmith()


class BaseModelFactory(ABC):
    """模型工厂抽象基类。"""

    @abstractmethod
    def generate(self) -> Optional[Union[Embeddings, BaseChatModel]]:
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂（OpenAI兼容版）"""

    def generate(self) -> Optional[BaseChatModel]:
        return ChatOpenAI(
            model=rag_conf["chat_model_name"],
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=rag_conf.get("temperature", 0.7),
            streaming=True,
        )


class EmbeddingsFactory(BaseModelFactory):
    """向量模型工厂（OpenAI兼容版）"""

    def generate(self) -> Optional[Embeddings]:
        return OpenAIEmbeddings(
            model=rag_conf["embedding_model_name"],
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )


def _require_openai_keys() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("缺少 OPENAI_API_KEY")

    if not os.getenv("OPENAI_BASE_URL"):
        raise EnvironmentError("缺少 OPENAI_BASE_URL")


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """惰性获取聊天模型，并通过缓存避免重复初始化。"""
    _require_openai_keys()
    return ChatModelFactory().generate()


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """惰性获取 embedding 模型，并通过缓存避免重复初始化。"""
    _require_openai_keys()
    return EmbeddingsFactory().generate()
