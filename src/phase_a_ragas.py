from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

# Điểm dưới ngưỡng này ở worst_metric mới tính là failure thật sự.
FAILURE_THRESHOLD = 0.7

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        # Dùng thẳng bottom_10() để report có cả diagnosis + suggested_fix
        "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved → {path}")


# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhom 50 cau hoi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups: dict[str, list[dict]] = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        dist = item.get("distribution", "factual")
        groups.setdefault(dist, []).append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chay RAGAS 4 metrics tren toan bo 50 cau hoi.

    Goi evaluate_ragas() tu src/m4_eval.py (Day 18) roi ghep voi distribution info.
    """
    if not answers:
        return []

    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("Khong tim thay src/m4_eval.py - da copy tu Day 18 chua?")
        return []

    questions     = [a["question"] for a in answers]
    ans_texts     = [a["answer"] for a in answers]
    contexts      = [a.get("contexts", []) for a in answers]
    ground_truths = [a.get("ground_truth", "") for a in answers]

    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = raw.get("per_question", []) if isinstance(raw, dict) else list(raw)

    def _get(obj, key: str) -> float:
        value = obj.get(key, 0.0) if isinstance(obj, dict) else getattr(obj, key, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return 0.0 if value != value else value  # NaN -> 0.0

    results: list[RagasResult] = []
    for a, pq in zip(answers, per_q):
        results.append(RagasResult(
            question_id=a.get("id", len(results) + 1),
            distribution=a.get("distribution", "factual"),
            question=a["question"],
            answer=a["answer"],
            contexts=a.get("contexts", []),
            ground_truth=a.get("ground_truth", ""),
            faithfulness=_get(pq, "faithfulness"),
            answer_relevancy=_get(pq, "answer_relevancy"),
            context_precision=_get(pq, "context_precision"),
            context_recall=_get(pq, "context_recall"),
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lay 10 cau hoi co avg_score thap nhat kem diagnosis + fix."""
    output: list[dict] = []
    for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10]):
        diagnosis, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank": i + 1,
            "question_id": r.question_id,
            "distribution": r.distribution,
            "question": r.question,
            "avg_score": round(r.avg_score, 4),
            "worst_metric": r.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult], threshold: float = FAILURE_THRESHOLD) -> dict:
    """Task 4: Phân tích failure clusters theo (worst_metric × distribution).

    Chỉ những câu thực sự fail (điểm của worst_metric < threshold) mới vào matrix —
    nếu đếm mọi câu thì mỗi câu luôn có một worst_metric và matrix chỉ phản ánh
    số câu của từng distribution chứ không phải chỗ hỏng.

    Dominant distribution chấm theo **tỉ lệ fail** (fail / tổng câu của distribution)
    để 10 câu adversarial không bị 20 câu factual lấn át.

    Returns:
        {"matrix": {metric: {dist: count}}, "dominant_failure_distribution": str,
         "dominant_failure_metric": str, "failure_rate_by_distribution": {...},
         "avg_score_by_distribution": {...}, "total_failures": int, "insight": str}
    """
    dists = ["factual", "multi_hop", "adversarial"]
    matrix = {metric: {d: 0 for d in dists} for metric in DIAGNOSTIC_TREE}

    for r in results:
        if getattr(r, r.worst_metric) < threshold and r.distribution in matrix[r.worst_metric]:
            matrix[r.worst_metric][r.distribution] += 1

    counts = {d: sum(1 for r in results if r.distribution == d) for d in dists}
    fails = {d: sum(matrix[m][d] for m in matrix) for d in dists}
    failure_rate = {d: round(fails[d] / counts[d], 4) for d in dists if counts[d]}
    avg_score = {
        d: round(sum(r.avg_score for r in results if r.distribution == d) / counts[d], 4)
        for d in dists if counts[d]
    }

    dominant_dist = max(failure_rate, key=failure_rate.get) if failure_rate else dists[0]
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    total_failures = sum(fails.values())

    insight = (
        f"{total_failures}/{len(results)} câu fail (worst_metric < {threshold}). "
        f"Distribution '{dominant_dist}' có tỉ lệ fail cao nhất "
        f"({fails[dominant_dist]}/{counts[dominant_dist]} = "
        f"{failure_rate.get(dominant_dist, 0):.0%}). "
        f"Metric '{dominant_metric}' hỏng nhiều nhất ({sum(matrix[dominant_metric].values())} câu). "
        f"Chẩn đoán: {DIAGNOSTIC_TREE[dominant_metric][0]}. "
        f"Fix: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )

    return {
        "matrix": matrix,
        "failure_threshold": threshold,
        "total_failures": total_failures,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "failure_rate_by_distribution": failure_rate,
        "avg_score_by_distribution": avg_score,
        "insight": insight,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            print(f"  #{item['rank']} [{item['distribution']}] {item['question'][:50]}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
    else:
        print("⚠️  No results — implement run_ragas_50q() first.")
