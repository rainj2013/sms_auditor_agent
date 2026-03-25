# 短信合规审核 Agent

## 用途

输入短信内容，自动识别短信类型（验证码 / 营销 / 催收 / 权益通知），基于合规规范进行多步推理（ReAct），输出结构化审核结果、违规原因及修改建议。

## 目录结构

项目分为两个独立实现：

```
sms_auditor/
├── vanilla/                    # 非 LangChain 版本
│   ├── sms_auditor_llm.py    # ReAct Agent（直接用 openai SDK）
│   ├── llm_providers.py       # LLM Provider 抽象层
│   ├── embeddings.py          # Embeddings 封装
│   └── rule_retriever.py      # ChromaDB 向量检索
│
├── langchain_agent/           # LangChain 版本
│   ├── sms_auditor.py        # Agent（用 langchain-openai + create_agent）
│   ├── minimax_embeddings.py  # LangChain Embeddings 抽象实现
│   └── rule_retriever.py      # langchain_chroma 向量检索
│
├── rules/                     # 合规规范文档
│   ├── 00_短信合规总纲.md
│   ├── 01_验证码短信规范.md
│   ├── 02_营销短信规范.md
│   ├── 03_催收短信规范.md
│   └── 04_权益通知短信规范.md
│
├── llm_config.json
└── README.md
```

## 快速开始

```bash
# 设置 API Key（当前默认 provider 为 minimax）
export MINIMAX_API_KEY=sk-xxxx

# 非 LangChain 版本
python3 vanilla/sms_auditor_llm.py "您的短信内容"

# LangChain 版本
python3 langchain_agent/sms_auditor.py "您的短信内容"
```

## 两种 Agent 对比

| 目录 | LLM 交互 | 主要依赖 |
|------|---------|---------|
| `vanilla/` | 直接用 `openai` SDK | `openai` |
| `langchain_agent/` | 用 `langchain-openai` + LangChain Agent | `langchain`, `langchain-openai`, `langchain-chroma` |

两个 Agent 审核逻辑相同，但 LLM 交互方式完全独立。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | LLM 提供商 | `minimax` |
| `KIMI_API_KEY` | Kimi API Key | — |
| `KIMI_MODEL` | Kimi 模型 | `moonshot-v1-128k` |
| `KIMI_BASE_URL` | Kimi API 地址 | `https://api.moonshot.cn/v1` |
| `MINIMAX_API_KEY` | MiniMax API Key | — |
| `MINIMAX_MODEL` | MiniMax 模型 | `MiniMax-M2.7-highspeed` |
| `LLM_TEMPERATURE` | 采样温度 | `1.0` |
| `LLM_MAX_TOKENS` | 最大输出 token 数 | `4096` |
| `LLM_TIMEOUT` | 请求超时（秒） | `120` |
| `LLM_DEBUG` | 流式输出（设为 `1` 开启） | `0` |

## 支持的 Provider

所有 Provider 均采用 OpenAI 标准格式（/v1/chat/completions），通过 `openai` SDK 调用：

- **kimi** — 月之暗面 Moonshot
- **minimax** — MiniMax
- **openai_compatible** — OpenAI / Claude / Gemini 等兼容接口

切换 Provider：`export LLM_PROVIDER=minimax`

## LLM_DEBUG 流式输出

（非 LangChain 版本）开启后 LLM 输出内容实时打印，减少等待感：

```bash
LLM_DEBUG=1 python3 vanilla/sms_auditor_llm.py "您的短信内容"
```

## 输出示例

```
============================================================
  📋 SMS 合规审核报告
============================================================
  短信类型 ：催收
  审核结果 ：🟡 整改
  判定理由 ：缺少必含信息（客服联系方式、还款方式、退订选项），且'征信黑名单'和'追究法律责任'表述不规范
------------------------------------------------------------
  📌 检查详情：
    1. ✅ [发送主体真实性] 签名【XX银行】与实际发送主体一致
    2. ✅ [恐吓威胁与伪造公文] 未发现一票否决项
    3. ✅ [隐私保护] 未泄露敏感信息
    4. ❌ [必含信息完整性] 缺少客服联系方式、还款方式、退订方式
    5. ❌ [用语文明与准确性] '征信黑名单'表述不规范
------------------------------------------------------------
  ✏️  修改后内容：
  【XX银行】您的贷款已逾期30天，欠款金额5000元。长期逾期将严重影响个人征信，并可能导致法律诉讼。请尽快通过APP还款或联系客服协商方案。客服热线：400-XXX-XXXX（工作日9:00-18:00），退订回T
============================================================
```
