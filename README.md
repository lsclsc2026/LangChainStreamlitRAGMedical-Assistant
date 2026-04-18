# 医疗分诊 + 心理咨询 GraphRAG 系统

<p align="center">
  <img src="docs/images/banner.png" alt="项目横幅" width="100%" />
</p>

> 一个面向中文场景的多模块智能对话系统：医疗分诊对话 + 心理咨询 GraphRAG。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 核心能力](#2-核心能力)
- [3. 效果预览](#3-效果预览)
- [4. 项目结构](#4-项目结构)
- [5. 环境准备](#5-环境准备)
- [6. 配置说明](#6-配置说明)
- [7. 快速运行](#7-快速运行)
- [8. 优化方向](#8-优化方向)
- [9. License](#9-license)
- [10. 联系方式](#10-联系方式)

---

## 1. 项目简介

本项目基于 Streamlit、LangChain、LangGraph、Chroma 构建，聚焦两类高价值应用：

- 医疗分诊对话模块：面向症状采集、风险分级、分层建议。
- 心理咨询模块（GraphRAG）：基于指定心理学资料进行全局/局部检索与支持性回答。

项目目标不是“单轮问答 Demo”，而是可持续演进的对话系统原型：支持多轮会话、上传记忆、知识库管理、流式反馈、后端 API 能力。

---

## 2. 核心能力

- 医疗分诊工作流：信息充足度判断 -> 严重程度评估 -> 路由检索 -> 建议生成。
- 心理咨询 GraphRAG：Global Search -> 模糊度判断 -> Local Search -> 支持回答。
- RAG 检索增强：向量召回 + 本地 reranker（可配置开启/路径）。
- OCR 能力：本地 OCR 与讯飞 PDF OCR 任务接口双链路。
- 多会话与记忆：会话持久化、上传文件摘要记忆、上下文整合。
- 管理员能力：知识库上传/删除/重建（前端 + FastAPI）。

---

## 3. 效果预览

### 4.1 医疗分诊页面（预置路径）

![医疗分诊页面](docs/images/ui-medical.png)

### 4.2 心理咨询页面（预置路径）

![心理咨询页面](docs/images/ui-psych.png)

### 4.3 流式流程展示（预置路径）

![流程流式展示](docs/images/ui-stream-trace.png)

---

## 4. 项目结构

```text
langchain-agent-master
├─app.py                                  # Streamlit 前端入口
├─backend/
│  └─api_server.py                        # FastAPI 后端接口（上传记忆、知识库管理）
├─agent/
│  ├─react_agent.py                       # 医疗分诊 Agent 入口
│  ├─triage_workflow.py                   # 医疗分诊核心工作流
│  ├─psych_consult_agent.py               # 心理咨询 Agent 入口
│  └─tools/
│     ├─agent_tools.py                    # 工具定义（RAG/天气/报告等）
│     └─middleware.py                     # 中间件逻辑
├─psych/
│  └─graphrag_service.py                  # 心理模块 GraphRAG 服务
├─rag/
│  ├─vector_store.py                      # 文档处理与向量入库
│  └─rag_service.py                       # 检索、重排、总结
├─model/
│  └─factory.py                           # Chat / Embedding 模型工厂
├─utils/
│  ├─config_handler.py                    # 配置加载
│  ├─prompt_loader.py                     # 提示词加载
│  ├─chat_session_store.py                # 会话持久化
│  ├─upload_memory_store.py               # 上传记忆持久化
│  └─xfyun_pdf_ocr_client.py              # 讯飞 PDF OCR 客户端
├─config/                                 # 配置目录（rag/chroma/prompt/agent/psych）
├─prompts/                                # 提示词目录
├─data/                                   # 知识库目录
├─storage/                                # 本地存储（会话/向量库/记忆）
└─docs/
   └─images/                              # README 图片目录
```

---

## 5. 环境准备

在项目根目录执行：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

建议使用 Python 3.10+ 与虚拟环境。

---

## 6. 配置说明

### 7.1 必需环境变量

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

可写入系统环境变量，或项目根目录 `.env`。

### 7.2 可选环境变量

- `ADMIN_TOKEN`：管理员鉴权。
- `AGENT_USER_CITY`：默认城市。
- `AGENT_USER_ID`：默认用户 ID。
- `XFYUN_OCR_APP_ID` / `XFYUN_OCR_API_SECRET`：覆盖心理模块 OCR 配置。

### 7.3 关键配置文件

- `config/rag.yaml`：模型名、温度、本地 reranker 路径等。
- `config/chroma.yaml`：切块和检索参数。
- `config/agent.yaml`：分诊性能参数（如 `triage_perf`）。
- `config/psych.yaml`：心理模块与 OCR 相关配置。


---

## 7. 快速运行


```bash
streamlit run app.py
```

---

## 8. 优化方向

当前系统的优化重点是把“可用”推进到“可运营”：一方面通过快慢路径分流、混合检索和多级缓存降低平均时延；另一方面通过节点级指标和回归评测保障质量稳定。对于面向真实用户的对话系统，能稳定控制 P95 时延、重复追问率、空召回率，通常比单次回答效果更关键。

---

## 9. License

本项目感谢并参考以下开源生态：

- Streamlit
- LangChain / LangGraph
- ChromaDB
- FastAPI
- sentence-transformers
- PyYAML / Requests / PyPDF

仅用于学习与技术参考，不可直接替代专业医疗诊断。

---

## 10. 联系方式

- 作者：李硕晨
- 邮箱：19051093016@163.com
- 电话：19051093016

---

## 附：图片资源命名建议

放在 `docs/images/` 下，建议使用以下命名，便于维护：

- `banner.png`
- `ui-medical.png`
- `ui-psych.png`
- `ui-stream-trace.png`
