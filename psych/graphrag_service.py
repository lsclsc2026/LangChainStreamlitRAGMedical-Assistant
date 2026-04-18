import json
import os
import re
from io import BytesIO
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import get_embedding_model
from utils.config_handler import psych_conf
from utils.file_handler import clean_text, pdf_loader
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.spark_ocr_client import SparkOcrClient
from utils.xfyun_pdf_ocr_client import XfyunPdfOcrClient

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None

try:
    import pypdfium2 as pdfium
except Exception:  # pragma: no cover
    pdfium = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
    RapidOCR = None


@dataclass
class GraphNode:
    node_id: str
    text: str
    source: str
    page: int | None
    keywords: list[str]


class PsychGraphRagService:
    """心理咨询专用 GraphRAG：只处理指定 PDF 并提供 global/local 检索。"""

    def __init__(self):
        self.pdf_relative_path = "data/伯恩斯新情绪疗法.pdf"
        self.pdf_abs_path = get_abs_path(self.pdf_relative_path)
        self.ocr_text_output_path = get_abs_path("data/心理咨询OCR提取文本.txt")
        self.persist_dir = get_abs_path("storage/psych_graph_db")
        self.graph_path = get_abs_path("storage/psych_graph.json")
        self.graph_artifacts_dir = get_abs_path("storage/psych_graph_artifacts")
        self.collection_name = "psych_graph_rag"
        self.chunk_size = 650
        self.chunk_overlap = 120
        self.use_cloud_ocr = bool(psych_conf.get("use_cloud_ocr", True))
        self.ocr_provider = psych_conf.get("ocr_provider", "xfyun_pdf_ocr")
        self.spark_batch_pages = int(psych_conf.get("spark_batch_pages", 1))
        self.fallback_to_local_ocr = bool(psych_conf.get("fallback_to_local_ocr", True))
        self.spark_ocr_client = SparkOcrClient()
        self.xfyun_pdf_ocr_client = XfyunPdfOcrClient()

        os.makedirs(self.persist_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        os.makedirs(self.graph_artifacts_dir, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=self.persist_dir,
            client_settings=Settings(anonymized_telemetry=False),
        )

        self.graph = {"nodes": {}, "edges": {}}
        self._ready = False

    def _save_extracted_text(self, docs: list[Document]) -> None:
        """把 OCR/抽取后的文本落盘到 data 目录，按固定文件覆盖保存。"""
        lines = []
        for idx, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_label = f"第{page + 1}页" if isinstance(page, int) else f"片段{idx}"
            lines.append(f"\n===== {page_label} =====\n")
            lines.append(clean_text(doc.page_content))

        final_text = "\n".join([line for line in lines if line])
        with open(self.ocr_text_output_path, "w", encoding="utf-8") as f:
            f.write(final_text)
        logger.info(f"心理咨询 OCR 文本已保存: {self.ocr_text_output_path}")

    @staticmethod
    def _is_text_sufficient(docs: list[Document]) -> bool:
        """判断常规 PDF 文本抽取是否足够，避免扫描版直接得到空库。"""
        if not docs:
            return False
        non_empty = 0
        total_chars = 0
        for d in docs:
            text = clean_text(d.page_content)
            if text:
                non_empty += 1
                total_chars += len(text)
        return non_empty >= 5 and total_chars >= 2000

    def _ocr_pdf_to_documents(self) -> list[Document]:
        """对扫描版 PDF 做本地 OCR，输出可切分的 Document 列表。"""
        if pdfium is None or RapidOCR is None:
            raise RuntimeError(
                "当前环境缺少 OCR 依赖，请安装 pypdfium2 和 rapidocr-onnxruntime。"
            )

        ocr_engine = RapidOCR()
        pdf = pdfium.PdfDocument(self.pdf_abs_path)
        ocr_docs: list[Document] = []

        for page_index in range(len(pdf)):
            page = pdf.get_page(page_index)
            try:
                # scale=2 兼顾中文识别率和处理速度。
                bitmap = page.render(scale=2).to_numpy()
                ocr_result, _ = ocr_engine(bitmap)
                if not ocr_result:
                    continue
                lines = []
                for item in ocr_result:
                    if not item or len(item) < 2:
                        continue
                    text = (item[1] or "").strip()
                    if text:
                        lines.append(text)
                merged = clean_text("\n".join(lines))
                if not merged:
                    continue
                ocr_docs.append(
                    Document(
                        page_content=merged,
                        metadata={
                            "source": os.path.basename(self.pdf_abs_path),
                            "page": page_index,
                            "ocr": True,
                        },
                    )
                )
            finally:
                page.close()

        return ocr_docs

    def _render_page_to_png_base64(self, page) -> str:
        if Image is None:
            raise RuntimeError("缺少 Pillow 依赖，无法将 PDF 页面转为 PNG。")
        bitmap = page.render(scale=2).to_numpy()
        image = Image.fromarray(bitmap)
        with BytesIO() as buf:
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        return self.spark_ocr_client.bytes_to_base64(image_bytes)

    def _cloud_ocr_pdf_to_documents(self) -> list[Document]:
        """使用云端 OCR 识别 PDF 页面。"""
        if pdfium is None:
            raise RuntimeError("缺少 pypdfium2，无法进行云端 OCR 页面渲染。")
        if not self.spark_ocr_client.is_configured():
            raise RuntimeError("Spark OCR 未配置 URL 或 API Key。")

        pdf = pdfium.PdfDocument(self.pdf_abs_path)
        docs: list[Document] = []
        for page_index in range(len(pdf)):
            page = pdf.get_page(page_index)
            try:
                b64 = self._render_page_to_png_base64(page)
                text = self.spark_ocr_client.ocr_page_base64(b64, page_index=page_index)
                text = clean_text(text)
                if not text:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(self.pdf_abs_path),
                            "page": page_index,
                            "ocr": True,
                            "ocr_provider": "spark_http",
                        },
                    )
                )
            finally:
                page.close()
        return docs

    def _xfyun_pdf_ocr_to_documents(self) -> list[Document]:
        """使用讯飞 PDF OCR 任务接口识别整份文档。"""
        if not self.xfyun_pdf_ocr_client.is_configured():
            raise RuntimeError("XFYUN PDF OCR 未配置：请填写 xfyun_app_id 与 xfyun_api_secret")

        task_no = self.xfyun_pdf_ocr_client.start_task(self.pdf_abs_path)
        data = self.xfyun_pdf_ocr_client.wait_until_finished(task_no)
        page_results = self.xfyun_pdf_ocr_client.fetch_result_text(data)

        docs: list[Document] = []
        for item in page_results:
            text = clean_text(item.get("text", ""))
            if not text:
                continue
            page_num = item.get("page")
            page_idx = int(page_num) - 1 if isinstance(page_num, int) and page_num > 0 else None
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(self.pdf_abs_path),
                        "page": page_idx,
                        "ocr": True,
                        "ocr_provider": "xfyun_pdf_ocr",
                    },
                )
            )
        return docs

    def _load_pdf_documents(self) -> list[Document]:
        """优先常规提取，文本不足时自动 OCR 回退。"""
        docs = pdf_loader(self.pdf_abs_path)
        for d in docs:
            d.page_content = clean_text(d.page_content)
            d.metadata["source"] = os.path.basename(self.pdf_abs_path)

        if self._is_text_sufficient(docs):
            logger.info("心理咨询 PDF 使用常规文本抽取")
            return [d for d in docs if d.page_content]

        logger.warning("检测到 PDF 文本不足，启动 OCR 流程")

        if self.use_cloud_ocr and self.ocr_provider == "xfyun_pdf_ocr":
            try:
                logger.info("优先使用讯飞 PDF OCR 任务接口")
                ocr_docs = self._xfyun_pdf_ocr_to_documents()
                if ocr_docs:
                    logger.info(f"讯飞 PDF OCR 提取完成，页文本数={len(ocr_docs)}")
                    return ocr_docs
                raise ValueError("讯飞 PDF OCR 返回空文本")
            except Exception as cloud_err:
                logger.warning(f"云端 OCR 失败: {str(cloud_err)}")
                if not self.fallback_to_local_ocr:
                    raise

        if self.use_cloud_ocr and self.ocr_provider == "spark_http":
            try:
                logger.info("优先使用星火云端 OCR")
                ocr_docs = self._cloud_ocr_pdf_to_documents()
                if ocr_docs:
                    logger.info(f"云端 OCR 提取完成，页文本数={len(ocr_docs)}")
                    return ocr_docs
                raise ValueError("云端 OCR 返回空文本")
            except Exception as cloud_err:
                logger.warning(f"云端 OCR 失败: {str(cloud_err)}")
                if not self.fallback_to_local_ocr:
                    raise

        logger.info("切换到本地 OCR 回退")
        ocr_docs = self._ocr_pdf_to_documents()
        if not ocr_docs:
            raise ValueError("OCR 未提取到有效文本，请检查 PDF 清晰度或更换 OCR 引擎。")
        logger.info(f"OCR 提取完成，页文本数={len(ocr_docs)}")
        return ocr_docs

    def _save_parquet_artifacts(self):
        """导出 GraphRAG 的节点和边为 parquet，便于后续离线分析与追溯。"""
        if pa is None or pq is None:
            logger.warning("未安装 pyarrow，跳过 parquet 工件导出")
            return

        nodes_rows = []
        for node_id, node in self.graph.get("nodes", {}).items():
            nodes_rows.append(
                {
                    "node_id": node_id,
                    "source": node.get("source"),
                    "page": node.get("page"),
                    "text": node.get("text", ""),
                    "keywords": "|".join(node.get("keywords") or []),
                }
            )

        edges_rows = []
        for src, targets in self.graph.get("edges", {}).items():
            for dst in targets:
                edges_rows.append({"src": src, "dst": dst})

        nodes_path = os.path.join(self.graph_artifacts_dir, "nodes.parquet")
        edges_path = os.path.join(self.graph_artifacts_dir, "edges.parquet")

        nodes_table = pa.Table.from_pylist(nodes_rows) if nodes_rows else pa.table({"node_id": [], "source": [], "page": [], "text": [], "keywords": []})
        edges_table = pa.Table.from_pylist(edges_rows) if edges_rows else pa.table({"src": [], "dst": []})

        pq.write_table(nodes_table, nodes_path)
        pq.write_table(edges_table, edges_path)
        logger.info(f"GraphRAG parquet工件已导出: {self.graph_artifacts_dir}")

    def _extract_keywords(self, text: str) -> list[str]:
        terms = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", text.lower())
        stop = {
            "我们", "你们", "他们", "自己", "这种", "这个", "那个", "可以", "需要", "以及", "因为",
            "然后", "如果", "但是", "所以", "焦虑", "恐惧", "症状", "治疗", "咨询", "心理",
        }
        uniq = []
        seen = set()
        for t in terms:
            if t in stop:
                continue
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq[:12]

    def _connect(self, a: str, b: str) -> None:
        if a == b:
            return
        self.graph["edges"].setdefault(a, [])
        if b not in self.graph["edges"][a]:
            self.graph["edges"][a].append(b)

    def _build_graph(self, docs) -> None:
        nodes = {}
        edges = {}
        self.graph = {"nodes": nodes, "edges": edges}

        last_id = ""
        for idx, doc in enumerate(docs):
            node_id = f"n_{idx}"
            content = clean_text(doc.page_content)
            keywords = self._extract_keywords(content)
            nodes[node_id] = {
                "text": content,
                "source": doc.metadata.get("source", os.path.basename(self.pdf_abs_path)),
                "page": doc.metadata.get("page"),
                "keywords": keywords,
            }
            edges.setdefault(node_id, [])
            if last_id:
                self._connect(last_id, node_id)
                self._connect(node_id, last_id)
            last_id = node_id

        # 基于关键词重叠补充语义边。
        node_items = list(nodes.items())
        for i in range(len(node_items)):
            node_a_id, node_a = node_items[i]
            set_a = set(node_a["keywords"])
            if not set_a:
                continue
            for j in range(i + 1, min(i + 25, len(node_items))):
                node_b_id, node_b = node_items[j]
                set_b = set(node_b["keywords"])
                overlap = set_a & set_b
                if len(overlap) >= 3:
                    self._connect(node_a_id, node_b_id)
                    self._connect(node_b_id, node_a_id)

        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(self.graph, f, ensure_ascii=False, indent=2)
        self._save_parquet_artifacts()

    def rebuild(self) -> None:
        if not os.path.exists(self.pdf_abs_path):
            raise FileNotFoundError(f"未找到心理咨询PDF: {self.pdf_abs_path}")

        docs = self._load_pdf_documents()
        self._save_extracted_text(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "，", " "],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
        chunks = [c for c in chunks if clean_text(c.page_content)]
        if not chunks:
            raise ValueError("GraphRAG 切分后为空，请确认 OCR/文本提取是否成功。")

        try:
            self.vector_store.delete_collection()
        except Exception:
            pass

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=get_embedding_model(),
            persist_directory=self.persist_dir,
            client_settings=Settings(anonymized_telemetry=False),
        )

        ids = []
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            ids.append(f"psych_{i}")

        batch = 20
        for i in range(0, len(chunks), batch):
            self.vector_store.add_documents(chunks[i:i + batch], ids=ids[i:i + batch])

        self._build_graph(chunks)
        self._ready = True
        logger.info(f"心理咨询 GraphRAG 重建完成，chunk数量={len(chunks)}")

    def ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            count = self.vector_store._collection.count()
        except Exception:
            count = 0
        if count <= 0 or not os.path.exists(self.graph_path):
            self.rebuild()
        else:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                self.graph = json.load(f)
            self._ready = True

    def global_search(self, query: str, k: int = 8):
        self.ensure_ready()
        return self.vector_store.similarity_search(query, k=k)

    def _score_node(self, query_terms: set[str], node: dict) -> float:
        node_terms = set(node.get("keywords") or [])
        overlap = len(query_terms & node_terms)
        coverage = overlap / max(1, len(query_terms))
        return coverage

    def local_search(self, query: str, seed_docs: list, max_nodes: int = 8) -> list[dict]:
        self.ensure_ready()
        query_terms = set(self._extract_keywords(query))

        seed_pages = set()
        for d in seed_docs:
            p = d.metadata.get("page")
            if isinstance(p, int):
                seed_pages.add(p)

        candidate_ids = []
        for node_id, node in self.graph.get("nodes", {}).items():
            if node.get("page") in seed_pages:
                candidate_ids.append(node_id)

        expanded = set(candidate_ids)
        for cid in list(candidate_ids):
            for n in self.graph.get("edges", {}).get(cid, []):
                expanded.add(n)

        scored = []
        for node_id in expanded:
            node = self.graph["nodes"].get(node_id)
            if not node:
                continue
            score = self._score_node(query_terms, node)
            scored.append((node_id, node, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        top_nodes = []
        for node_id, node, score in scored[:max_nodes]:
            top_nodes.append(
                {
                    "node_id": node_id,
                    "source": node.get("source"),
                    "page": node.get("page"),
                    "score": round(score, 4),
                    "text": node.get("text", ""),
                }
            )
        return top_nodes
