#!/usr/bin/env python3
"""
SMS 合规审核 Agent — 基于 LLM 的 ReAct 实现
输入短信内容，调用大模型进行多步推理，输出结构化审核结果
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

from vanilla.llm_providers import LLMMessage, get_provider
from vanilla.rule_retriever import RuleRetriever

# ─────────────────────────────────────────────
# 路径
# ─────────────────────────────────────────────

RULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules")
SYSTEM_PROMPT_PATH = os.path.join(RULES_DIR, "audit_system_prompt.md")


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
    overall: str          # 🟢🟡🔴
    level: str            # green/yellow/red
    reason: str
    checks: list[CheckItem]
    corrected_content: str = ""
    raw_response: str = ""


# ─────────────────────────────────────────────
# 规则加载
# ─────────────────────────────────────────────

def load_rules(sms_type: str | None = None) -> str:
    """加载合规规范文档

    - sms_type 为 None 时：只加载通用规则
    - sms_type 指定时：加载通用规则 + 对应类型的专项规则
    """
    rule_files = {
        "00_短信合规总纲.md": "通用规则（所有短信类型适用）",
        "01_验证码短信规范.md": "验证码短信规则",
        "02_营销短信规范.md": "营销短信规则",
        "03_催收短信规范.md": "催收短信规则",
        "04_权益通知短信规范.md": "权益通知短信规则",
    }

    type_map = {
        "验证码": "01_验证码短信规范.md",
        "营销": "02_营销短信规范.md",
        "催收": "03_催收短信规范.md",
        "权益通知": "04_权益通知短信规范.md",
    }

    content_lines = ["# 合规规范数据库\n"]

    # 通用规则（始终加载）
    filepath = os.path.join(RULES_DIR, "00_短信合规总纲.md")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        content_lines.append(f"\n## 通用规则（来源：00_短信合规总纲.md）\n")
        content_lines.append(text)

    # 类型专项规则（Round 2+ 时加载）
    if sms_type and sms_type in type_map:
        filename = type_map[sms_type]
        filepath = os.path.join(RULES_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            desc = filename.split("_", 1)[1].replace(".md", "")
            content_lines.append(f"\n## {sms_type}专项规则（来源：{filename}）\n")
            content_lines.append(text)

    return "\n".join(content_lines)


def identify_sms_type(sms: str) -> str:
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


# ─────────────────────────────────────────────
# System Prompt 构造
# ─────────────────────────────────────────────

def build_system_prompt(rules_text: str, round: int = 1) -> str:
    if round == 1:
        return f"""你是一名专业的金融行业短信合规审核员。

## 你的职责
收到短信内容和合规规则后，分阶段审核。第一阶段先根据通用规则判断短信类型并检查一票否决项。

## 合规规则全文
{rules_text}

## 第一轮任务
1. 根据短信内容判断类型（验证码/营销/催收/权益通知）
2. 检查通用红线：伪造主体、恐吓威胁、伪造法律文书、泄露隐私
3. 输出一句话结论，告知进入第二轮

## 审核原则（通用红线）
- 伪造发送主体：冒充银行/监管/政府 → 🔴一票否决
- 恐吓威胁：威胁人身安全、恐吓上门 → 🔴一票否决
- 伪造法律文书：冒充法院/公安 → 🔴一票否决
- 泄露隐私：明文展示身份证/完整银行卡 → 🔴一票否决

请开始第一轮审核。"""
    else:
        return f"""你是一名专业的金融行业短信合规审核员。

## 你的职责
第二阶段：结合专项规则完成最终审核，输出 JSON 结果。

## 合规规则全文
{rules_text}

## 输出格式（必须严格遵循）
最终必须输出一行 JSON，不得省略或改变字段名：
```
{{"sms_type":"...","overall":"...","level":"...","reason":"...","checks":[{{"category":"...","passed":true/false,"detail":"...","suggestion":"..."}}],"corrected_content":"..."}}
```

- sms_type：短信类型（验证码/营销/催收/权益通知/未知）
- overall：审核结果（🟢 合规 / 🟡 整改 / 🔴 违规）
- level：合规等级（green / yellow / red）
- reason：判定理由
- checks：检查项数组
- corrected_content：修改后内容

## 审核原则
1. 存在一票否决项直接判🔴违规
2. 违规内容需给出修改后的合规版本

## 合规等级说明
- 🟢 green：完全符合所有规范
- 🟡 yellow：存在需整改项（信息不完整/警告项）
- 🔴 red：存在违规内容或一票否决项

请输出最终 JSON 审核结果。"""


# ─────────────────────────────────────────────
# LLM 调用（ReAct 主循环）
# ─────────────────────────────────────────────

class ReActSMSAuditor:
    """
    ReAct SMS 审核 Agent
    每一轮由 LLM 自行推理（Thought→Action→Observation），直至给出最终 JSON
    """

    MAX_ITERATIONS = 3  # 最多推理轮数，防止无限循环

    def __init__(self, sms_content: str, provider=None, sms_type: str | None = None):
        self.sms = sms_content.strip()
        self.provider = provider
        self.sms_type = sms_type if sms_type else identify_sms_type(self.sms)
        self.reasoning_steps: list[str] = []
        self.retriever = RuleRetriever()

    def _build_messages(self, assistant_reasoning: str = "", round: int = 1) -> list[LLMMessage]:
        """构造对话消息

        - round=1：只加载通用规则，LLM 先判断类型并检查通用红线
        - round=2+：通过向量检索召回相关规则片段，输出最终 JSON
        """
        if round == 1:
            rules_text = load_rules(None)  # 只加载通用规则
        else:
            # 向量检索召回相关规则片段
            chunks = self.retriever.search(self.sms, k=8, sms_type=self.sms_type)
            rules_parts = [f"# 召回的专项规则（来源：向量检索）\n"]
            for i, (chunk, score) in enumerate(chunks, 1):
                rules_parts.append(
                    f"### [{i}] [{chunk.category}] {chunk.section}（相似度: {score:.4f}）\n{chunk.content}\n"
                )
            rules_text = "\n".join(rules_parts)

        system = build_system_prompt(rules_text, round=round)

        if round == 1:
            user = f"""## 待审核短信
```
{self.sms}
```

## 你的任务
1. 根据短信内容判断短信类型（验证码/营销/催收/权益通知/未知）
2. 对照通用规则检查是否存在一票否决项（伪造主体、恐吓威胁、伪造法律文书、泄露隐私等）
3. 如果没有一票否决项，进入第二轮加载专项规则继续审核
4. 第一轮请先输出推理过程（Thought/Action/Observation），然后明确告知"第一轮通用规则检查结束，请进入第二轮进行专项规则审核"，暂不输出最终 JSON。"""
        else:
            user = f"""## 待审核短信
```
{self.sms}
```

## 短信类型
类型：**{self.sms_type}**

## 你的任务
上方已通过向量检索召回与该短信最相关的专项规则片段，请对照审核：
1. 必含信息是否完整
2. 用语是否规范
3. 给出最终 JSON 审核结果。"""

        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ]

        if assistant_reasoning:
            messages.append(LLMMessage(role="assistant", content=assistant_reasoning))
            messages.append(LLMMessage(
                role="user",
                content="请继续推理，最终输出 JSON 结果。"
            ))

        return messages

    def _parse_json_output(self, text: str) -> Optional[AuditResult]:
        """从 LLM 输出中提取 JSON"""
        import re as _re

        # 策略1：从 ```json ... ``` 代码块中提取
        # 匹配 ```json（或仅 ```）后第一个 { 到第一个 }``` 之间的内容
        block_pattern = r"```(?:json)?[^\n]*\n*([\s\S]*?)\n*```"
        block_match = _re.search(block_pattern, text, _re.DOTALL)
        if block_match:
            candidate = block_match.group(1).strip()
            # 在代码块内容中找 JSON 对象（从第一个 { 到最后一个 }）
            json_match = _re.search(r"(\{[\s\S]*\})", candidate)
            if json_match:
                json_str = json_match.group(1)
                try:
                    data = json.loads(json_str)
                    return self._build_result(data, text)
                except json.JSONDecodeError:
                    pass  # 继续尝试其他策略

        # 策略2：全文搜索 JSON 对象 { ... }
        # 找最后一个（最外层）JSON 对象
        all_starts = [(m.start(), m.group()) for m in _re.finditer(r"\{", text)]
        all_ends = [(m.start(), m.group()) for m in _re.finditer(r"\}", text)]
        if all_starts and all_ends:
            # 从后往前找第一个有效的 JSON
            for end_idx in range(len(all_ends) - 1, -1, -1):
                end_pos = all_ends[end_idx][0]
                for start_idx in range(len(all_starts) - 1, -1, -1):
                    start_pos = all_starts[start_idx][0]
                    if start_pos < end_pos:
                        json_str = text[start_pos:end_pos + 1]
                        try:
                            data = json.loads(json_str)
                            return self._build_result(data, text)
                        except json.JSONDecodeError:
                            continue  # 尝试更小的范围
        return None

    def _build_result(self, data: dict, raw_response: str) -> AuditResult:
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

    def audit(self) -> AuditResult:
        """执行 ReAct 推理循环"""
        provider = self.provider or get_provider()
        model_name = getattr(provider, "model", "unknown")

        assistant_output = ""

        print(f"\n🔍 短信内容：{self.sms[:80]}{'...' if len(self.sms) > 80 else ''}")
        print(f"🤖 模型：{model_name}  |  类型：{self.sms_type}\n")
        print("─" * 60)

        for i in range(self.MAX_ITERATIONS):
            round_num = i + 1
            print(f"\n📝 推理轮次 {round_num}/{self.MAX_ITERATIONS}")
            print("─" * 40)

            messages = self._build_messages(assistant_output, round=round_num)

            temperature = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
            timeout = int(os.environ.get("LLM_TIMEOUT", "120"))
            max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))
            debug = os.environ.get("LLM_DEBUG", "0") == "1"

            stream_buffer = []

            def on_chunk(chunk):
                if debug:
                    print(chunk, end="", flush=True)
                stream_buffer.append(chunk)

            try:
                response = provider.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    stream=debug,
                    stream_callback=on_chunk,
                )
            except Exception as e:
                print(f"❌ API 调用失败：{e}")
                return AuditResult(
                    sms_type=self.sms_type,
                    overall="🟡 整改",
                    level="yellow",
                    reason=f"API 调用失败：{e}",
                    checks=[],
                    raw_response=str(e),
                )

            raw = response.content.strip()
            if debug:
                print()  # 换行，结束流式输出行
            else:
                print(f"\n📤 LLM 输出：\n{raw[:800]}{'...' if len(raw) > 800 else ''}")

            # 尝试解析 JSON
            result = self._parse_json_output(raw)
            if result:
                print("\n✅ 成功解析结构化结果")
                result.raw_response = raw
                # Round 1 有一票否决项（red）直接返回
                if round_num == 1 and result.level == "red":
                    return result
                # Round 1 其他结果，继续进入 Round 2 做专项审核
                if round_num == 1:
                    assistant_output = raw
                    self.reasoning_steps.append(raw)
                    continue
                return result

            # 未解析出 JSON，视为推理未完成，继续下一轮
            print("\n⚠️ 未检测到最终 JSON，继续推理...")
            assistant_output = raw
            self.reasoning_steps.append(raw)

        # 超过最大轮次
        return AuditResult(
            sms_type=self.sms_type,
            overall="🟡 整改",
            level="yellow",
            reason="推理超时，未能在规定轮次内得出结论",
            checks=[],
            corrected_content="",
            raw_response="\n".join(self.reasoning_steps),
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
    parser = argparse.ArgumentParser(description="SMS 合规审核 Agent")
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

    agent = ReActSMSAuditor(sms_content, sms_type=args.sms_type)
    result = agent.audit()
    print_result(result)


if __name__ == "__main__":
    main()
