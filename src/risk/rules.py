from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


ORDER_PATTERNS = [
    r"订单(号|编号)?",
    r"\b\d{12,}\b",
]

PHONE_PATTERNS = [
    r"1[3-9]\d{9}",
]

ADDRESS_PATTERNS = [
    r"(省|市|区|县|街道|小区|单元|门牌|收货地址)",
]

COMPENSATION_PATTERNS = [
    r"赔(我|偿)?\s*\d+",
    r"补偿\s*\d+",
    r"赔付\s*\d+",
    r"退差价\s*\d+",
    r"赔钱",
    r"赔偿金额",
]

COMPLAINT_PATTERNS = [
    r"投诉",
    r"曝光",
    r"起诉",
    r"律师",
    r"消协",
    r"12315",
    r"平台举报",
]

ABSOLUTE_PROMISE_PATTERNS = [
    r"一定退款",
    r"保证退款",
    r"保证送达",
    r"马上赔偿",
    r"立刻赔偿",
    r"百分百(解决|退款|赔偿)",
    r"肯定能退",
]

FAKE_LOGISTICS_PATTERNS = [
    r"已经到达.{0,8}(上海|北京|广州|深圳|杭州|成都|武汉|南京|重庆)",
    r"快递员正在派送",
    r"预计今天送达",
    r"已经出库",
]

MONEY_COMMITMENT_PATTERNS = [
    r"(赔偿|补偿|赔付|退还|返还).{0,8}\d+(\.\d+)?\s*元",
]


def _hit_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text or "") for p in patterns)


def pre_check_user_query(query: str) -> Dict[str, Any]:
    """
    生成前风控：判断用户问题是否属于高风险，需要人工介入或谨慎回答。
    """
    flags = []

    if _hit_any(query, ORDER_PATTERNS):
        flags.append("order_status_or_order_id")
    if _hit_any(query, PHONE_PATTERNS):
        flags.append("privacy_phone")
    if _hit_any(query, ADDRESS_PATTERNS):
        flags.append("privacy_address")
    if _hit_any(query, COMPENSATION_PATTERNS):
        flags.append("compensation_request")
    if _hit_any(query, COMPLAINT_PATTERNS):
        flags.append("complaint_escalation")

    need_human = bool(flags)

    return {
        "need_human": need_human,
        "risk_level": "high" if need_human else "low",
        "risk_flags": flags,
    }


def post_check_answer(answer: str) -> Dict[str, Any]:
    """
    生成后风控：检查模型输出是否有越界承诺或疑似编造。
    """
    flags = []

    if _hit_any(answer, ABSOLUTE_PROMISE_PATTERNS):
        flags.append("absolute_promise")
    if _hit_any(answer, FAKE_LOGISTICS_PATTERNS):
        flags.append("fake_logistics_status")
    if _hit_any(answer, MONEY_COMMITMENT_PATTERNS):
        flags.append("money_commitment")

    return {
        "has_risk": bool(flags),
        "risk_level": "high" if flags else "low",
        "risk_flags": flags,
    }


def retrieval_check(retrieved_docs: List[Dict[str, Any]], score_threshold: float = 0.35) -> Dict[str, Any]:
    """
    检索可信度检查。
    """
    if not retrieved_docs:
        return {
            "retrieval_reliable": False,
            "risk_flags": ["no_retrieved_docs"],
        }

    top1 = retrieved_docs[0]
    score = float(top1.get("score", 0.0) or 0.0)

    if score < score_threshold:
        return {
            "retrieval_reliable": False,
            "risk_flags": ["low_retrieval_score"],
            "top1_score": score,
        }

    return {
        "retrieval_reliable": True,
        "risk_flags": [],
        "top1_score": score,
    }


def make_safe_fallback(query: str, reason_flags: Optional[List[str]] = None) -> str:
    flags = set(reason_flags or [])

    if "compensation_request" in flags or "money_commitment" in flags:
        return (
            "理解您的诉求。关于具体赔付或补偿金额，我这边不能直接承诺，"
            "需要结合订单、商品和售后审核结果进一步确认。建议您联系人工客服处理，"
            "并提供订单信息和问题照片/凭证，以便尽快核实。"
        )

    if "order_status_or_order_id" in flags or "fake_logistics_status" in flags:
        return (
            "理解您想尽快确认订单或物流进度。由于我无法直接查询您的具体订单状态，"
            "建议您先在订单详情页查看最新物流信息；如果长时间未更新，"
            "可以联系人工客服并提供订单号，由客服协助核实。"
        )

    if "complaint_escalation" in flags:
        return (
            "非常抱歉给您带来不好的体验。您的问题建议转人工客服进一步处理，"
            "人工客服可以结合订单、沟通记录和售后规则为您核实并给出处理方案。"
        )

    if "no_retrieved_docs" in flags or "low_retrieval_score" in flags:
        return (
            "抱歉，目前检索到的信息不足以支持我给出准确结论。为避免误导您，"
            "建议您联系人工客服进一步核实，或补充商品名称、订单状态、问题截图等信息。"
        )

    return (
        "理解您的问题。为了避免给出不准确的信息，建议您联系人工客服进一步核实，"
        "客服会根据订单和售后规则协助处理。"
    )
