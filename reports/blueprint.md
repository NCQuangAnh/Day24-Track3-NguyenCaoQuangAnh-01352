# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Nguyễn Cao Quang Anh (01352)
**Ngày:** 2026-08-26

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (P50 25.7ms / P95 82.8ms — đo thực tế, Task 12)
[Presidio PII Scan]  (VN_CCCD, VN_PHONE, EMAIL, PHONE_NUMBER — whitelist pattern-based)
    │ block if: PII của cá nhân cụ thể xuất hiện trong query
    │ action:   return 400 + "PII detected in query"
    ▼ (P50 1123.8ms / P95 3219.1ms — LLM self-check của NeMo)
[Local pattern rail]  → chặn ngay, 0 LLM call
[NeMo Input Rail]     → self_check_input (gpt-4o-mini) cho phần còn lại
    │ block if: jailbreak / prompt injection / PII request / off-topic
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk (hierarchical) → M5 Enrich → M2 Hybrid (BM25+dense bge-m3) → M3 Rerank → gpt-4o-mini
    │ ~22s/query khi sinh 50 answers (1105.7s tổng)
    ▼
[NeMo Output Rail]  (local: PII scan trên answer + cụm nhạy cảm; + self_check_output)
    │ flag if:  PII trong response / nội dung nhạy cảm
    │ action:   redact hoặc thay bằng safe response
    ▼
User Response
```

---

## Guard Stack Pipeline

| Layer           | Tool          | Latency P95 | Failure Action |
|-----------------|---------------|-------------|----------------|
| PII Detection   | Presidio      | 82.76ms (target <10ms) | Reject + log |
| Topic/Jailbreak | Local pattern rail + NeMo Input | 3219.10ms (target <300ms) | 503 + reason |
| RAG Pipeline    | Day 18        | ~22.000ms (target <2000ms) | Fallback |
| Output Check    | Local PII/keyword + NeMo Output | ~3200ms (target <300ms) | Block + log |

---

## Latency Budget

*(`measure_p95_latency()` — Task 12, n_runs=10 trên 10 câu hỏi HR hợp lệ, tức input đi qua cả
hai layer. Nếu đo bằng adversarial input thì local rail chặn trước, NeMo không được gọi và số
đo ra ~0ms — không phản ánh chi phí thật.)*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget | Đạt? |
|---|---|---|---|---|---|
| Presidio PII | 25.74 | 82.76 | 82.76 | <10ms | (không đạt) |
| NeMo Input Rail | 1123.84 | 3219.10 | 3219.10 | <300ms | (không đạt) |
| RAG Pipeline | ~22.000 (1105.7s / 50 câu) | — | — | <2000ms | (không đạt) |
| NeMo Output Rail | ~1100 (cùng bậc input rail) | ~3200 | ~3200 | <300ms | (không đạt) |
| **Total Guard** | 1204.74 | **3236.40** | 3236.40 | **<500ms** | (không đạt) |

**Budget OK?** [ ] Yes / [x] No

**Comment:** Bottleneck là **NeMo input rail** — chiếm 3219/3236ms tức 99.5% tổng guard latency,
vì mỗi request là một LLM call `self_check_input` tới gpt-4o-mini. Presidio 82.76ms cũng vượt
budget 10ms do phải chạy spaCy `en_core_web_lg` pipeline trên mỗi text, dù bản thân regex
recognizer gần như miễn phí.

Kế hoạch đưa P95 về dưới 500ms:
1. **Giữ local pattern rail chạy trước NeMo** — hiện đã chặn 16/20 adversarial input với 0 LLM
   call, nên phần lớn traffic độc hại không phải trả 3s.
2. **Cache kết quả rail theo hash input** — câu hỏi HR lặp lại rất nhiều, cache hit đưa latency
   về ~0ms.
3. **Chạy Presidio song song với NeMo bằng `asyncio.gather`** thay vì tuần tự, tiết kiệm ~80ms.
4. **Bỏ NER khỏi Presidio**: đã whitelist entity pattern-based (`PII_ENTITIES`), bước tiếp là
   cấu hình `NlpEngine` rỗng để không load spaCy → Presidio về <10ms.
5. **Thay `self_check_input` bằng classifier local** (embedding + logistic regression) cho topic
   guard, chỉ escalate lên LLM khi classifier lưỡng lự.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75      # hiện tại: factual 0.933 (đạt) / multi_hop 0.496 (không đạt) / adversarial 0.533 (không đạt)
    MIN_AVG_SCORE: 0.65         # hiện tại: 0.885 / 0.731 / 0.649

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%) — hiện tại 20/20 (100%) (đạt)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms — hiện tại 3236ms (không đạt) (chưa cache, chưa bỏ spaCy)

- name: Judge Reliability Gate
  run: python src/phase_b_judge.py    # κ ≥ 0.6 — hiện tại 0.800 (đạt)

- name: Full test suite
  run: pytest tests/ -q               # 40/40 passed (đạt)
```

Trạng thái hiện tại: **2/4 gate chất lượng pass**. Gate RAGAS fail ở `multi_hop`/`adversarial`
(faithfulness < 0.75), gate latency fail vì chưa tối ưu NeMo.

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Bật cache rail / scale NeMo |
| Tỉ lệ request phải gọi NeMo (không bị local rail chặn) | > 60% | Kiểm tra pattern list đã lỗi thời |
| PII detected count | spike >10/hour | Security alert |
| Judge–human κ (re-label 20 câu/tuần) | < 0.6 | Ngưng dùng judge làm gate tự động |

---

## Monitoring (điền từ kết quả đo)

- **P95 latency thực tế:** 3236.40 ms (Presidio 82.76 + NeMo 3219.10) — vượt budget 500ms
- **Adversarial pass rate:** 20/20 (100%)
- **Worst RAGAS metric:** faithfulness (18/30 câu fail; multi_hop 0.496, adversarial 0.533)
- **Dominant failure distribution:** adversarial (tỉ lệ fail 9/10 = 90%)

### CI Gates (phải pass trước khi merge to main)
- [ ] RAGAS faithfulness ≥ 0.75 — factual 0.933 (đạt) nhưng multi_hop 0.496 (không đạt) / adversarial 0.533 (không đạt)
- [x] Adversarial suite pass rate ≥ 90% (18/20) — đạt 20/20
- [ ] P95 total guard latency < 500ms — đo được 3236ms (không đạt)

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | factual **0.885** / multi_hop **0.731** / adversarial **0.649** |
| Worst metric | **faithfulness** (18/30 câu fail rơi vào metric này) |
| Dominant failure distribution | **adversarial** — tỉ lệ fail 9/10 = 90% (multi_hop 75%, factual 30%) |
| Cohen's κ | **0.800** (`almost perfect`) — pointwise judge có reference, gpt-4o-mini |
| Position bias / Verbosity bias | 0.0 / 0.80 |
| Adversarial pass rate | **20 / 20** (100%) — pii 5/5, jailbreak 5/5, off_topic 5/5, prompt_injection 5/5 |
| Guard P95 latency | **3236 ms** (Presidio 82.76 + NeMo 3219.10) — vượt budget 500ms |

**Bonus đạt:** Phase A (adversarial 0.649 < factual 0.885), Phase B (κ 0.800 > 0.6), Phase C (20/20 ≥ 18/20).

---

## Nhận xét & Cải tiến

Layer rẻ nhất chặn được nhiều nhất: 16/20 adversarial input bị **local pattern rail** chặn với 0
LLM call, 4 câu còn lại bị Presidio chặn vì chứa CCCD/SĐT/email thật. NeMo `self_check_input`
đứng sau như lưới thứ hai. Đây là thứ tự đúng cho production — deterministic trước, LLM sau — và
nó là lý do pass rate 100% mà chi phí token gần như bằng 0 cho traffic tấn công.

Điểm yếu lớn nhất là **latency**: 3.2s P95 cho guard stack là không chấp nhận được, và 99.5% đến
từ một LLM call. Trong production tôi sẽ cache theo hash input + thay topic guard bằng classifier
local, chỉ giữ LLM cho các case biên.

Về chất lượng RAG: `faithfulness` của `multi_hop` chỉ 0.496 và `adversarial` 0.533, trong khi
`context_precision` ~0.95 ở cả ba nhóm. Nghĩa là **retrieval không phải thủ phạm — generation
mới là**. Pipeline lấy đúng chunk nhưng LLM trả lời vượt quá context: câu multi-hop cần cộng
trừ nhiều nguồn thì nó tự suy diễn con số, còn câu adversarial thì nó trộn v2023 với v2024. Fix
đúng chỗ là siết prompt ("chỉ dùng số có trong context, nêu rõ version, thiếu dữ liệu thì nói
thiếu"), hạ temperature, và gắn metadata `version`/`effective_date` để lọc bản hết hiệu lực
trước khi đưa vào context — chứ không phải tăng top-k.

LLM judge với κ = 0.800 đã đủ tin cậy để xếp hạng ứng viên answer, nhưng verbosity bias 0.80 cho
thấy nó vẫn ưu ái answer dài. Tôi sẽ giữ swap-and-average (position bias đo được 0.0 nhưng chi
phí chỉ gấp đôi call) và thêm ràng buộc súc tích vào prompt trước khi cho judge gác cổng deploy.

Một chi tiết cấu hình đáng ghi lại: `guardrails/rails.co` bản gốc dùng canonical form
(`user ask jailbreak`) làm input rail, nhưng với `nemoguardrails 0.23` các flow này **không chạy
LLM call nào** và `generate_async()` trả về chuỗi rỗng cho mọi input. Tôi đã thêm task
`self_check_input` / `self_check_output` vào `guardrails/config.yml` (bản gốc lưu ở
`config.yml.bak`) thì NeMo mới thực sự chặn — đó cũng là lúc latency thật 3.2s lộ ra.
