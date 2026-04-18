"""
总结服务类：将用户提问和参考资料给模型进行总结回复
"""
import json
import glob
import os
import re
import threading
from typing import Optional

from rag.vector_store import VectorStoreService
from utils.config_handler import chroma_conf, rag_conf
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_rag_prompts
from utils.logger_handler import logger
from langchain_core.prompts import PromptTemplate
from model.factory import get_chat_model
from langchain_core.output_parsers import StrOutputParser

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - 依赖缺失时降级到启发式重排
    CrossEncoder = None


class LocalReranker:
    """使用本地 CrossEncoder 对候选文档做重排序。"""

    _warned_missing_path = False
    _warned_missing_dependency = False

    def __init__(self):
        self.enabled = False
        self.model = None
        self.model_path = os.getenv("RERANKER_MODEL_PATH", "").strip() or rag_conf.get("local_reranker_model_path")
        self.model_id = os.getenv("RERANKER_MODEL_ID", "").strip() or rag_conf.get(
            "local_reranker_model_id", "Qwen/Qwen3-Reranker-0.6B"
        )
        self.cache_dir = os.getenv("RERANKER_CACHE_DIR", "").strip() or rag_conf.get(
            "local_reranker_cache_dir", ""
        )
        self.auto_download = str(
            os.getenv("RERANKER_AUTO_DOWNLOAD", str(rag_conf.get("local_reranker_auto_download", False)))
        ).lower() in {"1", "true", "yes", "y", "on"}
        self.device = rag_conf.get("local_reranker_device", "cpu")
        self.local_files_only = rag_conf.get("local_reranker_local_files_only", True)
        self._load_model()

    @staticmethod
    def _contains_model_files(path: str) -> bool:
        return os.path.isfile(os.path.join(path, "config.json"))

    def _resolve_snapshot_model_dir(self, base_dir: str) -> Optional[str]:
        """支持直接传 snapshots 根目录，自动定位包含 config.json 的快照目录。"""
        if not os.path.isdir(base_dir):
            return None
        if self._contains_model_files(base_dir):
            return base_dir

        snapshot_candidates = glob.glob(os.path.join(base_dir, "*"))
        snapshot_candidates = [p for p in snapshot_candidates if os.path.isdir(p)]
        for candidate in sorted(snapshot_candidates, reverse=True):
            if self._contains_model_files(candidate):
                return candidate
        return None

    def _resolve_model_path(self) -> Optional[str]:
        if not self.model_path:
            return None
        candidate = self.model_path
        if not os.path.isabs(candidate):
            candidate = get_abs_path(candidate)

        # 兼容传入 snapshots 根目录的情况。
        resolved = self._resolve_snapshot_model_dir(candidate)
        return resolved

    def _download_model(self) -> Optional[str]:
        if not self.auto_download:
            return None
        if CrossEncoder is None:
            return None
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            logger.warning("未安装 huggingface_hub，无法自动下载 reranker 模型")
            return None

        try:
            download_path = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir or None,
                local_files_only=bool(self.local_files_only),
            )
            resolved = self._resolve_snapshot_model_dir(download_path)
            if resolved:
                logger.info(f"已自动下载 reranker 模型到: {resolved}")
            return resolved
        except Exception as e:
            logger.warning(f"自动下载 reranker 模型失败: {str(e)}")
            return None

    def _load_model(self):
        resolved_model_path = self._resolve_model_path()

        if not resolved_model_path:
            if not LocalReranker._warned_missing_path:
                logger.warning(
                    "未找到可用 local reranker 模型目录，将使用启发式重排。"
                    "可在 config/rag.yaml 的 local_reranker_model_path 或环境变量 RERANKER_MODEL_PATH 配置。"
                )
                LocalReranker._warned_missing_path = True
            resolved_model_path = self._download_model()
            if not resolved_model_path:
                return

        if CrossEncoder is None:
            if not LocalReranker._warned_missing_dependency:
                logger.warning("未安装 sentence-transformers，将使用启发式重排")
                LocalReranker._warned_missing_dependency = True
            return
        try:
            self.model = CrossEncoder(
                resolved_model_path,
                device=self.device,
                automodel_args={"local_files_only": bool(self.local_files_only)},
            )
            self.enabled = True
            logger.info(f"本地重排序模型加载成功: {resolved_model_path} | device={self.device}")
        except Exception as e:
            logger.error(f"本地重排序模型加载失败: {str(e)}", exc_info=True)
            self.enabled = False

    def score(self, query: str, contents: list[str]) -> list[float]:
        if not contents:
            return []
        if not self.enabled or self.model is None:
            return [0.0 for _ in contents]
        pairs = [(query, content) for content in contents]
        try:
            scores = self.model.predict(pairs)
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"本地重排序打分失败: {str(e)}", exc_info=True)
            return [0.0 for _ in contents]

class RagSummarizeService(object):
    """RAG 服务入口，负责检索、重排、总结和来源整理。"""

    def __init__(self):
        """初始化向量库、提示词链路和检索相关参数。"""
        self.vector_store = VectorStoreService()
        self._collection_ready_checked = False
        self._repair_lock = threading.Lock()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = get_chat_model()
        self.chain = self._init_chain()
        self.top_k = chroma_conf["k"]
        self.candidate_k = chroma_conf.get("candidate_k", max(self.top_k * 2, self.top_k))
        self.min_relevance_score = chroma_conf.get("min_relevance_score", 0.0)
        self.local_reranker = LocalReranker()
        self.synonym_map = {
            "头疼": ["头痛", "头部疼痛", "颅部疼痛"],
            "头痛欲裂": ["剧烈头痛", "难以忍受头痛"],
            "发烧": ["发热", "体温升高", "高热"],
            "肌肉酸痛": ["全身酸痛", "肌痛"],
            "呕吐": ["恶心呕吐", "反胃"],
            "怕光": ["畏光", "光敏感"],
            "视力模糊": ["看不清", "视觉模糊"],
        }
        self.stopwords = {
            "的", "了", "呢", "吗", "呀", "啊", "我", "想", "请问", "一下", "怎么", "怎样",
            "是否", "一个", "这个", "那个", "可以", "需要", "有没有", "如何", "问题", "情况",
        }

    def _init_chain(self):
        """构造“提示词 -> 模型 -> 文本解析”的最小总结链。"""
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def _ensure_collection_ready(self):
        """
        向量库为空时自动触发一次本地知识入库，避免首次使用直接空检索。
        """
        if self._collection_ready_checked:
            return
        self._collection_ready_checked = True

        try:
            current_count = self.vector_store.vector_store._collection.count()
        except Exception as e:
            logger.error(f"获取向量库文档数量失败: {str(e)}", exc_info=True)
            return

        if current_count > 0:
            logger.info(f"当前向量库已有文档，数量: {current_count}")
            return

        logger.warning("检测到向量库为空，开始自动加载知识文档")
        try:
            self.vector_store.load_document()
            latest_count = self.vector_store.vector_store._collection.count()
            logger.info(f"自动加载完成，当前向量库文档数量: {latest_count}")
        except Exception as e:
            logger.error(f"自动加载知识文档失败: {str(e)}", exc_info=True)

    @staticmethod
    def _is_corrupted_index_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "hnsw segment reader" in message
            or "nothing found on disk" in message
            or "error executing plan" in message
        )

    def _repair_vector_store(self):
        """在检测到索引损坏时，串行重建向量库，避免并发修复。"""
        with self._repair_lock:
            logger.warning("检测到向量索引异常，开始重建向量库")
            self.vector_store.reset_store(clear_md5=True)
            self.vector_store.load_document(force_reload=True)
            self._collection_ready_checked = True
            latest_count = self.vector_store.get_collection_count()
            logger.info(f"向量库重建完成，当前文档数量: {latest_count}")

    @staticmethod
    def _normalize_query(query: str) -> str:
        """对用户问题做轻量规范化，统一一些常见别名。"""
        normalized = re.sub(r"\s+", " ", query.strip().lower())
        replacements = {
            "头疼": "头痛",
            "发烧": "发热",
            "怕光": "畏光",
            "看不清": "视力模糊",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)
        return normalized

    def _expand_query(self, query: str) -> str:
        """把口语化问题扩展成更适合检索的表达。"""
        normalized = self._normalize_query(query)
        expansions = []
        for phrase, candidates in self.synonym_map.items():
            if phrase in normalized:
                expansions.extend(candidates)
        if expansions:
            normalized = f"{normalized} {' '.join(expansions)}"
        return normalized

    def _query_terms(self, query: str) -> set[str]:
        """提取检索关键词，供后续重排计算覆盖率。"""
        expanded = self._expand_query(query)
        terms = set()
        for term in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", expanded):
            if term not in self.stopwords:
                terms.add(term)
        return terms

    @staticmethod
    def _document_terms(content: str) -> set[str]:
        """把文档内容切成词项集合，便于和 query 做简单交集比较。"""
        return set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", content.lower()))

    def _rerank_score(self, query_terms: set[str], content: str, relevance_score: float, local_score: float) -> float:
        """
        组合向量分数和关键词覆盖率。

        这里不是完整 reranker，而是一个成本很低的启发式重排，
        用来避免“向量相似但关键词没对上”的片段排得过高。
        """
        doc_terms = self._document_terms(content)
        overlap = len(query_terms & doc_terms)
        coverage = overlap / max(len(query_terms), 1)
        return relevance_score * 0.45 + coverage * 0.15 + local_score * 0.40

    @staticmethod
    def _match_source_keywords(source: str, source_keywords: Optional[list[str]]) -> bool:
        if not source_keywords:
            return True
        source_lower = source.lower()
        return any(keyword.lower() in source_lower for keyword in source_keywords)

    @staticmethod
    def _distance_to_relevance(distance_score: float) -> float:
        """把向量距离转换为 [0,1] 的相关度，数值越大表示越相关。"""
        score = 1.0 / (1.0 + max(float(distance_score), 0.0))
        return max(0.0, min(1.0, score))

    def retriever_docs(
        self,
        query,
        source_keywords: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        use_local_reranker: bool = True,
        candidate_k_override: Optional[int] = None,
    ):
        """执行检索主流程：查询扩展 -> 候选召回 -> 轻量重排 -> 截断返回。"""
        self._ensure_collection_ready()
        expanded_query = self._expand_query(query)
        query_terms = self._query_terms(query)
        final_top_k = top_k or self.top_k
        candidate_k = candidate_k_override or (self.candidate_k * 4 if source_keywords else self.candidate_k)
        try:
            raw_candidates = self.vector_store.vector_store.similarity_search_with_score(
                expanded_query,
                k=candidate_k,
            )
            candidates = [
                (doc, self._distance_to_relevance(distance_score))
                for doc, distance_score in raw_candidates
            ]
        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}", exc_info=True)
            if self._is_corrupted_index_error(e):
                try:
                    self._repair_vector_store()
                    raw_candidates = self.vector_store.vector_store.similarity_search_with_score(
                        expanded_query,
                        k=self.candidate_k,
                    )
                    candidates = [
                        (doc, self._distance_to_relevance(distance_score))
                        for doc, distance_score in raw_candidates
                    ]
                except Exception as repair_error:
                    logger.error(f"重建后检索仍失败: {str(repair_error)}", exc_info=True)
                    return []
            else:
                return []

        if use_local_reranker:
            local_scores = self.local_reranker.score(
                expanded_query,
                [doc.page_content for doc, _ in candidates],
            )
        else:
            local_scores = [0.0 for _ in candidates]

        # 先保留候选，再按综合分数重排。
        scored_docs = []
        for (doc, relevance_score), local_score in zip(candidates, local_scores):
            if relevance_score < self.min_relevance_score:
                continue
            source = doc.metadata.get("source", "")
            if not self._match_source_keywords(source, source_keywords):
                continue
            rerank_score = self._rerank_score(query_terms, doc.page_content, relevance_score, local_score)
            doc.metadata["relevance_score"] = round(float(relevance_score), 4)
            doc.metadata["local_rerank_score"] = round(float(local_score), 4)
            doc.metadata["rerank_score"] = round(float(rerank_score), 4)
            scored_docs.append((doc, rerank_score))

        scored_docs.sort(key=lambda item: item[1], reverse=True)
        docs = [doc for doc, _ in scored_docs[:final_top_k]]
        logger.info(
            f"RAG检索完成，原始query={query}，扩展query={expanded_query}，候选数={len(candidates)}，入选数={len(docs)}"
        )
        return docs

    def query_knowledge(self, query: str, knowledge_name: str, top_k: Optional[int] = None) -> str:
        """按知识库名称定向检索并返回拼接文本。"""
        docs = self.retriever_docs(
            query,
            source_keywords=[knowledge_name],
            top_k=top_k,
            use_local_reranker=False,
            candidate_k_override=max((top_k or self.top_k) * 3, top_k or self.top_k),
        )
        if not docs:
            return ""
        blocks = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            blocks.append(f"[{knowledge_name}-{index}] 来源={source}\n{doc.page_content.strip()}")
        return "\n\n".join(blocks)

    def query_multiple_knowledges(self, query: str, knowledge_names: list[str], top_k: int = 2) -> dict[str, str]:
        """批量按知识库名称检索，返回每个库的上下文文本。"""
        results = {}
        for name in knowledge_names:
            results[name] = self.query_knowledge(query, name, top_k=top_k)
        return results

    @staticmethod
    def _format_references(docs) -> str:
        """把命中的来源整理成回答尾部可展示的引用列表。"""
        references = []
        seen = set()
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            ref = f"{source} 第{page + 1}页" if isinstance(page, int) else source
            if ref not in seen:
                seen.add(ref)
                references.append(ref)
        if not references:
            return ""
        return "\n参考来源：\n- " + "\n- ".join(references)

    def rag_summarize(self, query):
        """对外暴露的 RAG 总入口，返回“总结结果 + 引用来源”。"""
        try:
            context_docs = self.retriever_docs(query)
        except Exception as e:
            logger.error(f"RAG检索流程异常: {str(e)}", exc_info=True)
            return "知识库检索暂时不可用，请稍后重试。"

        if not context_docs:
            return "未检索到相关参考资料。"

        # 把命中文档拼成可追踪来源的上下文，便于模型总结时引用。
        context_parts = []
        for counter, doc in enumerate(context_docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            chunk_index = doc.metadata.get("chunk_index")
            location_parts = [f"来源={source}"]
            if page is not None:
                location_parts.append(f"页码={page}")
            if chunk_index is not None:
                location_parts.append(f"切片={chunk_index}")
            context_parts.append(
                f"[参考资料{counter}] {' | '.join(location_parts)}\n{doc.page_content.strip()}"
            )
        context = "\n\n".join(context_parts)
        try:
            answer = self.chain.invoke(
                {
                    "input": query,
                    "context": context,
                }
            )
            return answer.strip() + self._format_references(context_docs)
        except Exception as e:
            logger.error(f"RAG总结失败: {str(e)}", exc_info=True)
            return "知识总结暂时不可用，请稍后重试。"


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合什么扫地机器人"))
