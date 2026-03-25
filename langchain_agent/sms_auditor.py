#!/usr/bin/env python3
"""
SMS 合规审核 Agent — LangChain 版本
使用 LangChain 原生 Agent 框架、VectorStore Retriever
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_agent.rule_retriever import get_vectorstore, search_rules


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class CheckItem:
    category: str
    passed: bool
    detail: str
    suggestion: str = ""


@dataclass
class AuditResult:
    sms_type: str
    overall: str
    level: str
    reason: str
    checks: list[CheckItem]
    corrected_content: str = ""
    raw_response: str = ""


# ─────────────────────────────────────────────
# LangChain Tool：向量检索召回专项规则
# ─────────────────────────────────────────────

@tool
def retrieve_rules(sms_content: str, sms_type: str = "") -> str:
    """
    向量检索召回专项规则片段，供审核使用。
    当第一轮审核结束后需要召回专项规则时调用此工具。

    Args:
        sms_content: 待审核的短信全文
        sms_type: 识别出的短信类型（验证码/营销/催收/权益通知），同类型规则优先
    Returns:
        召回的规则列表文本
    """
    chunks = search_rules(sms_content, k=8, sms_type=sms_type)

    if not chunks:
        return "未召回任何规则片段。"

    parts = []
    for i, (doc, score) in enumerate(chunks, 1):
        parts.append(
            f"[{i}] 类型:{doc.metadata.get('category')} | 章节:{doc.metadata.get('section')} | 相似度:{score:.4f}\n{doc.page_content}"
        )
    return "\n---\n".join(parts)


# ─────────────────────────────────────────────
# 提示词模板
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """你是一名专业的金融行业短信合规审核员。

## 你的职责
收到短信内容后，分两阶段审核：
1. 第一阶段（Round 1）：根据通用规则判断短信类型并检查一票否决项
2. 第二阶段（Round 2）：使用 retrieve_rules 工具召回专项规则，逐项审核

## 通用红线（一票否决项）
以下情况直接返回 red 级别违规，不进入第二轮：
- 伪造发送主体：冒充银行/监管/政府
- 恐吓威胁：威胁人身安全、恐吓上门
- 伪造法律文书：冒充法院/公安
- 泄露隐私：明文展示身份证/完整银行卡
- 保证收益承诺："保证收益"、"稳赚不赔"
- 嵌入钓鱼链接：未经安全审核的第三方链接

## 短信类型
- 验证码：含"验证码"、"动态密码"、"登录验证码"
- 营销：含"限时"、"优惠"、"活动"、"返现"、"加息"、"红包"、"新用户"、"年化收益"、"理财"、"投资"、"购买"
- 催收：含"逾期"、"催收"、"欠款"、"还款"、"诉讼"、"律师函"
- 权益通知：含"登录"、"交易"、"账户"、"到账"、"变更"、"通知"、"积分"、"会员"
- 未知：无法归类

## 可用工具
- retrieve_rules(sms_content, sms_type): 召回短信专项规则，返回规则文本

## 输出格式（Round 2 结束后必须输出）
最终必须输出以下 JSON 结构（不要包裹 markdown 代码块，直接输出 JSON 文本）：
{"sms_type":"...","overall":"...","level":"...","reason":"...","checks":[{"category":"...","passed":true/false,"detail":"...","suggestion":"..."}],"corrected_content":"..."}

字段说明：
- sms_type：短信类型（验证码/营销/催收/权益通知/未知）
- overall：审核结果（🟢 合规 / 🟡 整改 / 🔴 违规）
- level：合规等级（green / yellow / red）
- reason：判定理由（1-2句话）
- checks：检查项数组，每项包含 category/passed/detail/suggestion
- corrected_content：修改后的合规内容（无修改则为空字符串）

## 合规等级说明
- 🟢 green：完全符合所有规范
- 🟡 yellow：存在需整改项（信息不完整/警告项）
- 🔴 red：存在违规内容或一票否决项

## 推理流程
1. 先根据短信内容判断类型
2. 检查是否存在一票否决项
3. 如无否决项，使用 retrieve_rules 工具召回专项规则
4. 对照专项规则逐项检查
5. 输出最终 JSON 结果

最多进行 3 轮推理（防止无限循环）。
"""


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _identify_sms_type(sms: str) -> str:
    """根据关键词识别短信类型"""
    if any(k in sms for k in ["验证码", "动态密码", "登录验证码"]):
        return "验证码"
    if any(k in sms for k in ["逾期", "催收", "欠款", "还款", "诉讼", "律师函"]):
        return "催收"
    if any(k in sms for k in ["限时", "优惠", "活动", "返现", "加息", "红包", "新用户", "年化收益", "理财", "投资", "购买"]):
        return "营销"
    if any(k in sms for k in ["登录", "交易", "账户", "到账", "变更", "通知", "积分", "会员"]):
        return "权益通知"
    return "未知"


def _parse_json_output(text: str) -> Optional[AuditResult]:
    """从 LLM 输出中提取 JSON"""
    block_pattern = r"```(?:json)?[^\n]*\n*([\s\S]*?)\n*```"
    block_match = re.search(block_pattern, text, re.DOTALL)
    if block_match:
        candidate = block_match.group(1).strip()
        json_match = re.search(r"(\{[\s\S]*\})", candidate)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return _build_result(data, text)
            except json.JSONDecodeError:
                pass

    all_starts = [(m.start(), m.group()) for m in re.finditer(r"\{", text)]
    all_ends = [(m.start(), m.group()) for m in re.finditer(r"\}", text)]
    if all_starts and all_ends:
        for end_idx in range(len(all_ends) - 1, -1, -1):
            end_pos = all_ends[end_idx][0]
            for start_idx in range(len(all_starts) - 1, -1, -1):
                start_pos = all_starts[start_idx][0]
                if start_pos < end_pos:
                    json_str = text[start_pos:end_pos + 1]
                    try:
                        data = json.loads(json_str)
                        return _build_result(data, text)
                    except json.JSONDecodeError:
                        continue
    return None


def _build_result(data: dict, raw_response: str) -> AuditResult:
    """从解析出的 dict 构建 AuditResult"""
    return AuditResult(
        sms_type=data.get("sms_type", "未知"),
        overall=data.get("overall", "🟡 整改"),
        level=data.get("level", "yellow"),
        reason=data.get("reason", ""),
        checks=[
            CheckItem(
                category=c.get("category", ""),
                passed=c.get("passed", False),
                detail=c.get("detail", ""),
                suggestion=c.get("suggestion", ""),
            )
            for c in data.get("checks", [])
        ],
        corrected_content=data.get("corrected_content", ""),
        raw_response=raw_response,
    )


# ─────────────────────────────────────────────
# Agent 执行
# ─────────────────────────────────────────────

def run_audit(sms_content: str) -> AuditResult:
    """执行 LangChain Agent 审核，返回 AuditResult"""
    sms = sms_content.strip()
    sms_type = _identify_sms_type(sms)

    # 从环境变量获取配置
    provider_name = os.environ.get("LLM_PROVIDER", "minimax")
    if provider_name == "kimi":
        model = os.environ.get("KIMI_MODEL", "moonshot-v1-128k")
        base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
        api_key = os.environ.get("KIMI_API_KEY", "")
    elif provider_name == "minimax":
        model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7-highspeed")
        base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
        api_key = os.environ.get("MINIMAX_API_KEY", "")
    else:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "")

    print(f"\n🔍 短信内容：{sms[:80]}{'...' if len(sms) > 80 else ''}")
    print(f"🤖 模型：{model}  |  初始类型：{sms_type}\n")
    print("─" * 60)

    # 初始化 LangChain ChatOpenAI
    llm = ChatOpenAI(
        model=model,
        openai_api_base=base_url,
        api_key=api_key,
        temperature=1.0,
        max_tokens=4096,
    )

    # 创建 Agent
    tools = [retrieve_rules]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    # 构建输入
    user_input = (
        f"请审核以下短信，判断类型并检查合规性：\n{sms}\n\n"
        f"注意：如果第一轮检测到一票否决项（red级别），直接输出最终JSON结果。\n"
        f"如果没有否决项，请使用 retrieve_rules 工具召回专项规则继续审核，最终输出 JSON 结果。"
    )

    # 执行 Agent
    output = ""
    try:
        inputs = {"messages": [{"role": "user", "content": user_input}]}
        for chunk in agent.stream(inputs, stream_mode="values"):
            messages = chunk.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.content:
                    output = msg.content
    except Exception as e:
        return AuditResult(
            sms_type=sms_type,
            overall="🟡 整改",
            level="yellow",
            reason=f"Agent 执行失败：{e}",
            checks=[],
            corrected_content="",
            raw_response=str(e),
        )

    print(f"\n📤 Agent 输出：\n{output[:800]}{'...' if len(output) > 800 else ''}")

    # 解析 JSON
    parsed = _parse_json_output(output)
    if parsed:
        print("\n✅ 成功解析结构化结果")
        return parsed

    return AuditResult(
        sms_type=sms_type,
        overall="🟡 整改",
        level="yellow",
        reason="Agent 未返回有效 JSON 结果",
        checks=[],
        corrected_content="",
        raw_response=output,
    )


# ─────────────────────────────────────────────
# 格式化输出
# ─────────────────────────────────────────────

def print_result(result: AuditResult):
    """格式化打印最终结果"""
    print("\n" + "=" * 62)
    print(f"  📋 SMS 合规审核报告")
    print("=" * 62)
    print(f"  短信类型 ：{result.sms_type}")
    print(f"  审核结果 ：{result.overall}")
    print(f"  判定理由 ：{result.reason}")
    print("-" * 62)

    if result.checks:
        print("  📌 检查详情：")
        for i, check in enumerate(result.checks, 1):
            icon = "✅" if check.passed else "❌"
            print(f"    {i}. {icon} [{check.category}] {check.detail}")
            if check.suggestion:
                print(f"       💡 {check.suggestion}")

    if result.corrected_content:
        print("-" * 62)
        print("  ✏️  修改后内容：")
        print(f"  {result.corrected_content}")

    print("=" * 62)


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SMS 合规审核 Agent（LangChain 版）")
    parser.add_argument("sms", nargs="*", help="短信内容（可省略，交互式输入）")
    parser.add_argument("-t", "--type", dest="sms_type", help="短信类型（验证码/营销/催收/权益通知）")
    args = parser.parse_args()

    if args.sms:
        sms_content = " ".join(args.sms)
    else:
        print("\n📨 请输入待审核的短信内容（输入空行结束）：")
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break
        sms_content = "\n".join(lines)

    if not sms_content.strip():
        print("⚠️ 未检测到短信内容，请提供短信文本。")
        sys.exit(1)

    result = run_audit(sms_content)
    print_result(result)


if __name__ == "__main__":
    main()