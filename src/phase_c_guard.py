from __future__ import annotations

"""Phase C: Production Guardrails — Presidio PII + NeMo Guardrails + P95 Latency."""

import asyncio
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR, LATENCY_BUDGET_P95_MS,
                    PRESIDIO_LANGUAGE, TEST_SET_PATH)


# Chỉ nhận các entity dựa trên pattern/checksum. spaCy NER chạy model tiếng Anh
# (en_core_web_lg) nên gắn nhãn PERSON sai hàng loạt cho tiếng Việt ("nghỉ phép" -> PERSON),
# vì vậy các entity dựa trên NER bị loại khỏi whitelist.
PII_ENTITIES = [
    "VN_CCCD", "VN_PHONE", "EMAIL_ADDRESS", "PHONE_NUMBER",
    "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS", "US_SSN",
]

_PRESIDIO_ENGINES = None       # cache (analyzer, anonymizer)
_NEMO_RAILS = None             # cache LLMRails
_NEMO_UNAVAILABLE = False      # True nếu NeMo không import/khởi tạo được


# ─── Fallback regex PII (dùng khi Presidio chưa cài) ─────────────────────────

_PII_PATTERNS = [
    ("VN_CCCD",  re.compile(r"\b\d{12}\b"), 0.9),
    ("VN_CCCD",  re.compile(r"\b\d{9}\b"), 0.7),
    ("VN_PHONE", re.compile(r"\b0[3-9]\d{8}\b"), 0.9),
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), 0.9),
]


def _regex_pii_scan(text: str) -> dict:
    """Fallback PII scan bằng regex thuần khi Presidio không có sẵn.

    Cùng contract với pii_scan(): has_pii / entities / anonymized.
    """
    entities = []
    taken: list[tuple[int, int]] = []
    for entity_type, pattern, score in _PII_PATTERNS:
        for m in pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue  # đã bị recognizer ưu tiên hơn chiếm chỗ
            taken.append((m.start(), m.end()))
            entities.append({
                "type": entity_type, "text": m.group(),
                "score": score, "start": m.start(), "end": m.end(),
            })

    if not entities:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = text
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        anonymized = anonymized[:e["start"]] + f"<{e['type']}>" + anonymized[e["end"]:]
    entities.sort(key=lambda x: x["start"])
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9a: Presidio PII Detection ─────────────────────────────────────────

def setup_presidio():
    """Khởi tạo Presidio engine với custom Vietnamese PII recognizers. (Đã implement sẵn)

    Custom recognizers thêm vào:
        VN_CCCD  — số CCCD 12 chữ số hoặc CMND 9 chữ số
        VN_PHONE — số điện thoại Việt Nam (0[3-9]xxxxxxxx)

    Các recognizers mặc định đã có sẵn: EMAIL, PHONE_NUMBER (international), ...
    """
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, Pattern, PatternRecognizer
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits",  r"\b\d{9}\b",  0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer  = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def get_presidio():
    """Lấy (analyzer, anonymizer) đã cache; trả (None, None) nếu Presidio chưa cài."""
    global _PRESIDIO_ENGINES
    if _PRESIDIO_ENGINES is None:
        try:
            _PRESIDIO_ENGINES = setup_presidio()
        except Exception as exc:
            print(f"[warn] Presidio không khả dụng ({type(exc).__name__}) — dùng regex fallback.")
            _PRESIDIO_ENGINES = (None, None)
    return _PRESIDIO_ENGINES


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: Quét PII trong văn bản bằng Presidio (fallback regex nếu chưa cài).

    Returns:
        {
          "has_pii":    bool,
          "entities":   [{"type": str, "text": str, "score": float, "start": int, "end": int}],
          "anonymized": str,   # text với PII được thay bằng <TYPE>
        }
    """
    if analyzer is None or anonymizer is None:
        analyzer, anonymizer = get_presidio()
    if analyzer is None or anonymizer is None:
        return _regex_pii_scan(text)

    try:
        results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE,
                                   entities=PII_ENTITIES)
    except Exception as exc:
        print(f"[warn] Presidio analyze lỗi ({type(exc).__name__}) — dùng regex fallback.")
        return _regex_pii_scan(text)

    if not results:
        return {"has_pii": False, "entities": [], "anonymized": text}

    anonymized = anonymizer.anonymize(text=text, analyzer_results=results).text
    entities = [
        {"type": r.entity_type, "text": text[r.start:r.end],
         "score": round(r.score, 3), "start": r.start, "end": r.end}
        for r in sorted(results, key=lambda r: r.start)
    ]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


# ─── Task 9b + 11: NeMo Guardrails ───────────────────────────────────────────

def setup_nemo_rails():
    """Khởi tạo NeMo Guardrails từ guardrails/config.yml. (Đã implement sẵn)

    Config directory: guardrails/
        config.yml  — model + rails config
        rails.co    — Colang dialogue flows (topic check, jailbreak check, output check)
    """
    from nemoguardrails import RailsConfig, LLMRails
    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    rails  = LLMRails(config)
    return rails


def get_nemo_rails():
    """Lấy LLMRails đã cache; trả None nếu NeMo/API key không khả dụng."""
    global _NEMO_RAILS, _NEMO_UNAVAILABLE
    if _NEMO_UNAVAILABLE:
        return None
    if _NEMO_RAILS is None:
        if not os.getenv("OPENAI_API_KEY"):
            print("[warn] Không có OPENAI_API_KEY — NeMo rail dùng local pattern fallback.")
            _NEMO_UNAVAILABLE = True
            return None
        try:
            _NEMO_RAILS = setup_nemo_rails()
        except Exception as exc:
            print(f"[warn] NeMo Guardrails không khả dụng ({type(exc).__name__}) — "
                  "dùng local pattern fallback.")
            _NEMO_UNAVAILABLE = True
            return None
    return _NEMO_RAILS


REFUSE_KEYWORDS = [
    "xin lỗi", "không thể", "không được phép", "không thể cung cấp",
    "i cannot", "i can't", "i'm sorry", "i am sorry", "tôi không thể",
]

# Local pattern rail — cùng ý đồ với guardrails/rails.co, chạy được offline.
_JAILBREAK_PATTERNS = [
    "bỏ qua tất cả hướng dẫn", "bỏ qua hướng dẫn", "bỏ qua tất cả policy", "bỏ qua policy",
    "ignore all previous", "ignore your previous", "ignore previous instructions",
    "ignore your system prompt", "forget your system prompt", "forget all",
    "system override", "admin command", "do anything now", "dan (do anything now)",
    "pretend you are", "đóng vai", "không có giới hạn", "unrestricted ai",
    "no restrictions", "jailbreak", "developer mode",
    "in ra toàn bộ system", "print all confidential", "dump all",
    "tiết lộ", "mật khẩu admin", "tấn công mạng", "ra lệnh bạn bỏ qua",
    "ignore previous instructions",
]

_PII_REQUEST_PATTERNS = [
    "cccd của", "số điện thoại của nhân viên", "thông tin cá nhân của",
    "email của nhân viên", "lương của nhân viên", "lương tháng của nhân viên",
    "bảng lương chi tiết", "employee salaries", "employee records",
    "toàn bộ thông tin nhân viên", "confidential hr data", "employee data",
    "liệt kê toàn bộ thông tin",
]

_OFF_TOPIC_PATTERNS = [
    "bài thơ", "làm thơ", "nấu", "công thức truyền thống", "phở",
    "bitcoin", "ethereum", "giá cổ phiếu", "crypto",
    "bộ phim", "phim hay", "marvel", "giải phương trình", "phương trình vi phân",
    "dy/dx", "thời tiết", "bóng đá", "du lịch",
]

_HR_TOPIC_PATTERNS = [
    "nghỉ phép", "ngày phép", "phép năm", "bảo hiểm", "lương", "thưởng",
    "tạm ứng", "công tác", "thử việc", "đào tạo", "vpn", "mật khẩu",
    "wfh", "làm việc từ xa", "phụ cấp", "mentor", "nghỉ ốm", "chi phí",
    "mua sắm", "hiệu suất", "kỷ luật",
]


def _local_input_rail(text: str) -> dict:
    """Fallback rail offline: pattern check jailbreak / prompt injection / off-topic.

    Thứ tự ưu tiên giống rails.co: jailbreak → PII request → off-topic.
    Câu hỏi HR hợp lệ (chỉ khớp HR patterns) luôn được cho qua.
    """
    lowered = text.lower()

    def _hit(patterns: list[str]) -> str | None:
        for p in patterns:
            if p in lowered:
                return p
        return None

    hit = _hit(_JAILBREAK_PATTERNS)
    if hit:
        return {"allowed": False, "reason": f"jailbreak/prompt_injection: '{hit}'",
                "response": "Xin lỗi, tôi không thể thực hiện yêu cầu này."}

    hit = _hit(_PII_REQUEST_PATTERNS)
    if hit:
        return {"allowed": False, "reason": f"pii_request: '{hit}'",
                "response": "Xin lỗi, tôi không thể cung cấp thông tin cá nhân của nhân viên."}

    hit = _hit(_OFF_TOPIC_PATTERNS)
    if hit and not _hit(_HR_TOPIC_PATTERNS):
        return {"allowed": False, "reason": f"off_topic: '{hit}'",
                "response": "Xin lỗi, tôi chỉ trả lời câu hỏi về chính sách nội bộ công ty."}

    return {"allowed": True, "reason": None, "response": ""}


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: Kiểm tra input qua NeMo input rails (topic guard + jailbreak guard).

    Returns:
        {
          "allowed":        bool,
          "blocked_reason": str | None,
          "response":       str,          # NeMo's raw response
        }
    """
    # Layer nhanh, deterministic — chạy trước để tiết kiệm LLM call.
    local = _local_input_rail(text)
    if not local["allowed"]:
        return {"allowed": False,
                "blocked_reason": f"local_pattern_rail ({local['reason']})",
                "response": local["response"]}

    if rails is None:
        rails = get_nemo_rails()
    if rails is None:
        return {"allowed": True, "blocked_reason": None, "response": ""}

    try:
        response = await rails.generate_async(messages=[{"role": "user", "content": text}])
    except Exception as exc:
        print(f"[warn] NeMo generate lỗi ({type(exc).__name__}) — giữ kết quả local rail.")
        return {"allowed": True, "blocked_reason": None, "response": ""}

    if isinstance(response, dict):
        response = response.get("content", "")
    response = str(response)

    blocked = any(kw in response.lower() for kw in REFUSE_KEYWORDS)
    return {
        "allowed": not blocked,
        "blocked_reason": "nemo_input_rail" if blocked else None,
        "response": response,
    }


_SENSITIVE_OUTPUT_PATTERNS = [
    "cccd của nhân viên là", "số điện thoại cá nhân của", "mật khẩu hệ thống là",
    "thông tin bí mật", "bảng lương chi tiết",
]


def _local_output_rail(answer: str) -> dict:
    """Fallback output rail: PII trong answer + cụm nhạy cảm."""
    lowered = answer.lower()
    for p in _SENSITIVE_OUTPUT_PATTERNS:
        if p in lowered:
            return {"safe": False, "reason": f"sensitive_output: '{p}'"}

    pii = pii_scan(answer)
    if pii["has_pii"]:
        types = sorted({e["type"] for e in pii["entities"]})
        return {"safe": False, "reason": f"pii_in_response: {', '.join(types)}",
                "redacted": pii["anonymized"]}
    return {"safe": True, "reason": None}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: Kiểm tra LLM output qua NeMo output rails trước khi trả về user.

    Returns:
        {
          "safe":           bool,
          "flagged_reason": str | None,
          "final_answer":   str,          # answer đã qua guard (có thể bị redact)
        }
    """
    local = _local_output_rail(answer)
    if not local["safe"]:
        final = local.get("redacted") or (
            "Tôi không thể cung cấp thông tin này. Vui lòng liên hệ phòng Nhân sự trực tiếp.")
        return {"safe": False, "flagged_reason": local["reason"], "final_answer": final}

    if rails is None:
        rails = get_nemo_rails()
    if rails is None:
        return {"safe": True, "flagged_reason": None, "final_answer": answer}

    try:
        response = await rails.generate_async(messages=[
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ])
    except Exception as exc:
        print(f"[warn] NeMo output rail lỗi ({type(exc).__name__}) — giữ kết quả local rail.")
        return {"safe": True, "flagged_reason": None, "final_answer": answer}

    if isinstance(response, dict):
        response = response.get("content", "")
    response = str(response)

    flagged = any(kw in response.lower() for kw in REFUSE_KEYWORDS)
    return {
        "safe": not flagged,
        "flagged_reason": "nemo_output_rail" if flagged else None,
        "final_answer": response if flagged else answer,
    }


# ─── Task 10: Adversarial Test Suite ─────────────────────────────────────────

def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                          analyzer=None, anonymizer=None) -> list[dict]:
    """Task 10: Chạy 20 adversarial inputs qua full guard stack, so sánh với expected.

    Guard stack order:
        1. pii_scan()         → block nếu has_pii
        2. check_input_rail() → block nếu jailbreak / off-topic / prompt injection

    Returns:
        list of {"id", "category", "input", "expected", "actual", "blocked_by", "passed"}
    """
    if not adversarial_set:
        return []

    async def _run_all() -> list[dict]:
        out = []
        for item in adversarial_set:
            text = item["input"]
            blocked_by = None
            reason = None

            # Layer 1: Presidio PII (đồng bộ, nhanh)
            pii = pii_scan(text, analyzer, anonymizer)
            if pii["has_pii"]:
                blocked_by = "presidio"
                reason = "pii: " + ", ".join(sorted({e["type"] for e in pii["entities"]}))

            # Layer 2: NeMo input rail (await — không gọi asyncio.run() trong loop)
            if blocked_by is None:
                rail_result = await check_input_rail(text, rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"
                    reason = rail_result["blocked_reason"]

            actual = "blocked" if blocked_by else "allowed"
            out.append({
                "id":         item["id"],
                "category":   item["category"],
                "input":      text[:80] + "...",
                "expected":   item["expected"],
                "actual":     actual,
                "blocked_by": blocked_by,
                "blocked_reason": reason,
                "passed":     actual == item["expected"],
            })
        return out

    results = asyncio.run(_run_all())   # một lần duy nhất
    passed = sum(1 for r in results if r["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


# ─── Task 12: P95 Latency Measurement ────────────────────────────────────────

def _percentiles(times: list[float]) -> dict:
    """P50/P95/P99 theo nearest-rank trên list đã sort."""
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(times)
    n = len(s)

    def pick(q: float) -> float:
        idx = min(int(q * n), n - 1)
        return round(s[idx], 2)

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                        rails=None, analyzer=None, anonymizer=None) -> dict:
    """Task 12: Đo P50/P95/P99 latency cho từng layer trong guard stack.

    Returns:
        {"presidio_ms": {...}, "nemo_ms": {...}, "total_ms": {...},
         "latency_budget_ok": bool, "budget_ms": int}
    """
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for text in test_inputs[:n_runs]:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(presidio_ms + nemo_ms)

    asyncio.run(_measure())   # một lần duy nhất

    total_p = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms":     _percentiles(nemo_times),
        "total_ms":    total_p,
        "latency_budget_ok": total_p["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def save_phase_c_report(adv_results: list[dict], latency: dict, pii_demo: dict,
                        output_demo: dict | None = None,
                        path: str = "reports/guard_results.json") -> None:
    """Save Phase C report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    passed = sum(1 for r in adv_results if r["passed"])
    by_category: dict[str, dict] = {}
    for r in adv_results:
        entry = by_category.setdefault(r["category"], {"total": 0, "passed": 0})
        entry["total"] += 1
        entry["passed"] += int(r["passed"])

    report = {
        "adversarial_suite": {
            "total": len(adv_results),
            "passed": passed,
            "pass_rate": round(passed / len(adv_results), 3) if adv_results else 0.0,
            "by_category": by_category,
            "results": adv_results,
        },
        "latency": latency,
        "pii_demo": pii_demo,
        "output_rail_demo": output_demo,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Task 9a: PII scan demo
    test_pii = "Nhân viên Nguyễn Văn A, CCCD 034095001234, SĐT 0987654321 hỏi về nghỉ phép."
    pii_demo = pii_scan(test_pii)
    print(f"PII detected: {pii_demo['has_pii']}")
    print(f"Entities: {pii_demo['entities']}")
    print(f"Anonymized: {pii_demo['anonymized']}")

    # Task 10: Adversarial suite
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)
    print(f"\nLoaded {len(adversarial_set)} adversarial inputs")
    results = run_adversarial_suite(adversarial_set)
    for r in results:
        flag = "OK " if r["passed"] else "FAIL"
        print(f"  [{flag}] #{r['id']} {r['category']:<17} {r['actual']:<8} "
              f"by={r['blocked_by']}")

    # Task 11: Output rail demo
    output_demo = asyncio.run(check_output_rail(
        "Cho tôi số điện thoại của anh Nam phòng kế toán",
        "Số điện thoại cá nhân của anh Nam là 0912345678.",
    ))
    print(f"\nOutput rail — safe={output_demo['safe']} "
          f"reason={output_demo['flagged_reason']}")
    print(f"  final_answer: {output_demo['final_answer']}")

    # Task 12: P95 latency — đo trên câu hỏi HR hợp lệ để input đi hết cả 2 layer.
    # (Adversarial input bị local rail chặn ngay nên không chạm tới NeMo, đo sẽ ra ~0ms.)
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        sample_inputs = [item["question"] for item in json.load(f)[:10]]
    latency = measure_p95_latency(sample_inputs, n_runs=10)
    print(f"\nLatency P95 — Presidio: {latency['presidio_ms']['p95']}ms | "
          f"NeMo: {latency['nemo_ms']['p95']}ms | "
          f"Total: {latency['total_ms']['p95']}ms")
    print(f"Budget OK ({latency['budget_ms']}ms): {latency['latency_budget_ok']}")

    save_phase_c_report(results, latency, pii_demo, output_demo)
