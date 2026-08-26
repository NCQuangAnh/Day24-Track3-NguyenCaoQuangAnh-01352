from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH, TEST_SET_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

PROMPT_TEMPLATE = """Bạn là expert đánh giá chất lượng câu trả lời RAG cho trợ lý chính sách nhân sự.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác, độ đầy đủ, tính súc tích.
Bỏ qua thứ tự xuất hiện của hai answer — chỉ đánh giá nội dung.
Trả lời JSON (chỉ JSON, không text khác):
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "giải thích ngắn gọn", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}
"""


def _heuristic_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Fallback offline khi không có OPENAI_API_KEY hoặc gọi API thất bại.

    Chấm điểm bằng overlap từ vựng với câu hỏi + sự hiện diện số liệu + độ dài hợp lý.
    Không thay thế LLM judge trong production — chỉ để test/CI chạy được offline.
    """
    def _tokens(text: str) -> set[str]:
        return {t for t in text.lower().split() if len(t) > 2}

    q_tokens = _tokens(question)

    def _score(answer: str) -> float:
        a_tokens = _tokens(answer)
        if not a_tokens:
            return 0.0
        overlap = len(q_tokens & a_tokens) / max(len(q_tokens), 1)
        digits = 0.15 if any(ch.isdigit() for ch in answer) else 0.0
        length_fit = 1.0 - min(abs(len(answer) - 120) / 400.0, 1.0)
        return round(min(0.6 * overlap + digits + 0.35 * length_fit, 1.0), 3)

    score_a, score_b = _score(answer_a), _score(answer_b)
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
    else:
        winner = "A" if score_a > score_b else "B"
    return {
        "winner": winner,
        "reasoning": (f"[offline heuristic] score A={score_a}, B={score_b} — dựa trên overlap "
                      f"với câu hỏi, sự hiện diện số liệu, và độ dài."),
        "scores": {"A": score_a, "B": score_b},
    }


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    if not OPENAI_API_KEY:
        return _heuristic_judge(question, answer_a, answer_b)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as exc:  # lỗi network/key/parse không được làm vỡ pipeline
        fallback = _heuristic_judge(question, answer_a, answer_b)
        fallback["reasoning"] = f"[fallback: {type(exc).__name__}] " + fallback["reasoning"]
        return fallback

    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"

    raw_scores = parsed.get("scores") or {}
    scores = {}
    for key in ("A", "B"):
        try:
            scores[key] = max(0.0, min(1.0, float(raw_scores.get(key, 0.0))))
        except (TypeError, ValueError):
            scores[key] = 0.0

    reasoning = str(parsed.get("reasoning", "")).strip() or "Judge không trả về reasoning."
    return {"winner": winner, "reasoning": reasoning, "scores": scores}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

SWAP_MAP = {"A": "B", "B": "A", "tie": "tie"}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Pass 1: judge(q, A, B) — Pass 2: judge(q, B, A) rồi convert về không gian A/B gốc.
    Hai pass đồng ý → final = winner đó; khác nhau → final = "tie" (inconclusive).
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP

    winner_pass2 = SWAP_MAP.get(pass2_raw["winner"], "tie")
    position_consistent = pass1["winner"] == winner_pass2
    final = pass1["winner"] if position_consistent else "tie"

    raw2 = pass2_raw.get("scores", {}) or {}
    scores_pass2 = {"A": raw2.get("B", 0.0), "B": raw2.get("A", 0.0)}

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    κ = (p_o - p_e) / (1 - p_e), với p_o = observed agreement,
    p_e = expected agreement by chance. Trả về giá trị trong [-1, 1].
    """
    n = len(judge_labels)
    if n == 0 or n != len(human_labels):
        return 0.0

    p_o = sum(1 for j, h in zip(judge_labels, human_labels) if j == h) / n

    categories = set(judge_labels) | set(human_labels)
    p_e = sum((judge_labels.count(k) / n) * (human_labels.count(k) / n) for k in categories)

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    kappa = (p_o - p_e) / (1.0 - p_e)
    return max(-1.0, min(1.0, kappa))


def kappa_interpretation(kappa: float) -> str:
    """Thang Landis-Koch cho giá trị κ."""
    if kappa < 0:
        return "poor"
    if kappa < 0.2:
        return "slight"
    if kappa < 0.4:
        return "fair"
    if kappa < 0.6:
        return "moderate"
    if kappa < 0.8:
        return "substantial"
    return "almost perfect"


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo position bias và verbosity bias của judge."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0,
                                  "total_decisive": 0},
            "interpretation": "Chưa có kết quả judge nào.",
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate = position_bias_count / total

    a_wins_a_longer = sum(1 for r in judge_results
                          if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b))
    b_wins_b_longer = sum(1 for r in judge_results
                          if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a))
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = ((a_wins_a_longer + b_wins_b_longer) / decisive) if decisive else 0.0

    if position_bias_rate > 0.3:
        interpretation = ("Position bias cao (>30%) — bắt buộc dùng swap-and-average, "
                          "case không nhất quán nên coi là tie.")
    elif position_bias_rate > 0.1:
        interpretation = "Position bias trung bình — vẫn nên giữ swap-and-average."
    else:
        interpretation = "Position bias thấp — judge ổn định theo thứ tự."
    if verbosity_bias > 0.6:
        interpretation += (f" Verbosity bias {verbosity_bias:.0%} > 60% — judge ưu tiên answer "
                           "dài hơn, nên thêm ràng buộc súc tích vào prompt.")

    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Helpers: κ vs human labels + report ─────────────────────────────────────

POINTWISE_TEMPLATE = """Bạn là expert kiểm định câu trả lời của trợ lý chính sách nhân sự.

Câu hỏi: {question}

Đáp án chuẩn (ground truth):
{ground_truth}

Câu trả lời của model cần chấm:
{answer}

Chấm 1 nếu câu trả lời khớp đáp án chuẩn về mọi con số, ngưỡng, cấp phê duyệt và phiên bản
chính sách (thiếu một ý bắt buộc, sai số liệu, hoặc dùng phiên bản đã hết hiệu lực đều là 0).
Chấm 0 nếu sai hoặc thiếu ý bắt buộc.
Trả lời JSON (chỉ JSON): {{"label": 0 hoặc 1, "reasoning": "giải thích ngắn gọn"}}
"""


def judge_label_single(question: str, answer: str, ground_truth: str) -> int:
    """Chấm pointwise 0/1 cho một answer, có reference ground_truth.

    Dùng để đối chiếu với human labels (Task 7). Khác pairwise_judge() ở chỗ
    có đáp án chuẩn, nên bắt được lỗi sai số liệu / sai version chính sách.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là expert kiểm định RAG. Chỉ trả lời JSON."},
                    {"role": "user", "content": POINTWISE_TEMPLATE.format(
                        question=question, answer=answer, ground_truth=ground_truth)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return 1 if int(json.loads(resp.choices[0].message.content).get("label", 0)) == 1 else 0
        except Exception as exc:
            print(f"[warn] pointwise judge lỗi ({type(exc).__name__}) — dùng heuristic.")

    # Fallback offline: overlap số liệu giữa answer và ground truth
    import re as _re
    nums_gt = set(_re.findall(r"\d+", ground_truth))
    nums_ans = set(_re.findall(r"\d+", answer))
    if not nums_gt:
        return 1
    return 1 if len(nums_gt & nums_ans) / len(nums_gt) >= 0.5 else 0


def judge_labels_from_human_set(human_data: list[dict],
                                test_set_path: str = TEST_SET_PATH) -> list[int]:
    """Chấm 0/1 cho từng model_answer trong human_labels_10q.json bằng pointwise judge.

    Ground truth lấy từ test_set_50q.json theo question_id — judge có reference nên
    phân biệt được lỗi sai số liệu và sai version chính sách (v2023 vs v2024).
    """
    with open(test_set_path, encoding="utf-8") as f:
        ground_truths = {item["id"]: item["ground_truth"] for item in json.load(f)}

    return [
        judge_label_single(item["question"], item["model_answer"],
                           ground_truths.get(item["question_id"], ""))
        for item in human_data
    ]


def save_phase_b_report(judge_results: list[JudgeResult], kappa: float, bias: dict,
                        path: str = "reports/judge_results.json") -> None:
    """Save Phase B report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "judge_model": JUDGE_MODEL,
        "used_llm_api": bool(OPENAI_API_KEY),
        "total_pairs": len(judge_results),
        "cohen_kappa": round(kappa, 4),
        "kappa_interpretation": kappa_interpretation(kappa),
        "bias_report": bias,
        "pairs": [
            {
                "question": r.question,
                "answer_a": r.answer_a,
                "answer_b": r.answer_b,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
            }
            for r in judge_results
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

DEMO_PAIRS = [
    ("Nhân viên được nghỉ bao nhiêu ngày phép năm?",
     "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành.",
     "Theo quy định, nhân viên có 12 ngày phép hàng năm."),
    ("Mức phụ cấp ăn trưa là bao nhiêu?",
     "Phụ cấp ăn trưa là 50.000 VND/ngày làm việc thực tế.",
     "Công ty có hỗ trợ ăn trưa cho nhân viên theo quy định hiện hành."),
    ("Nhân viên thử việc bao lâu?",
     "Thời gian thử việc tối đa 60 ngày với vị trí chuyên môn, hưởng 85% lương.",
     "Thử việc thường kéo dài vài tháng tuỳ vị trí."),
    ("Quy trình tạm ứng lương như thế nào?",
     "Nhân viên nộp đơn tạm ứng cho quản lý trực tiếp, tối đa 50% lương tháng, "
     "duyệt trong 2 ngày làm việc.",
     "Bạn cần liên hệ phòng Nhân sự để được hướng dẫn tạm ứng."),
    ("Chính sách làm việc từ xa quy định gì?",
     "Nhân viên được WFH tối đa 2 ngày/tuần, đăng ký trước với quản lý.",
     "Công ty cho phép làm việc từ xa trong một số trường hợp."),
]

if __name__ == "__main__":
    # --- Task 5 + 6: pairwise + swap-and-average trên các cặp demo ---
    print("Running swap-and-average judge...")
    judge_results = []
    for q, a_a, a_b in DEMO_PAIRS:
        r = swap_and_average(q, a_a, a_b)
        judge_results.append(r)
        print(f"  [{q[:40]}...] pass1={r.winner_pass1} pass2={r.winner_pass2} "
              f"final={r.final_winner} consistent={r.position_consistent}")

    # --- Task 7: Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    judge_labels = judge_labels_from_human_set(human_data)
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Judge labels: {judge_labels}")
    print(f"Human labels: {human_labels}")
    print(f"Cohen's κ: {kappa:.3f} ({kappa_interpretation(kappa)})")

    # --- Task 8: bias report ---
    bias = bias_report(judge_results)
    print(f"\nPosition bias rate: {bias['position_bias_rate']} "
          f"({bias['position_bias_count']}/{bias['total_judged']})")
    print(f"Verbosity bias:     {bias['verbosity_bias']}")
    print(f"Interpretation:     {bias['interpretation']}")

    save_phase_b_report(judge_results, kappa, bias)
