# Failure Cluster Analysis — Phase A

**Sinh viên:** Nguyễn Cao Quang Anh (01352)
**Ngày:** 2026-08-26
**Nguồn số liệu:** `reports/ragas_50q.json` (RAGAS 0.1.22, judge model gpt-4o-mini,
pipeline Day 18: hierarchical chunk + enrich + hybrid BM25/bge-m3 + rerank)

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual (20q) | multi_hop (20q) | adversarial (10q) |
|---|---|---|---|
| faithfulness | **0.933** | 0.496 | 0.533 |
| answer_relevancy | 0.790 | 0.672 | **0.437** |
| context_precision | 0.958 | 0.942 | 0.958 |
| context_recall | 0.858 | 0.804 | 0.683 |
| **avg_score** | **0.885** | 0.731 | **0.649** |

Điểm mấu chốt: `context_precision` cao đều ~0.95 ở cả ba nhóm, nhưng `faithfulness` sụp xuống
0.50–0.53 ở `multi_hop` và `adversarial`. **Retriever lấy đúng chunk; generation mới là chỗ hỏng.**

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | Nhân viên thử việc tháng thứ 3 phát hiện vi phạm bảo mật… | 0.250 | faithfulness |
| 2 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.312 | faithfulness |
| 3 | multi_hop | Manager thâm niên 12 năm: tổng phụ cấp hàng tháng… | 0.375 | faithfulness |
| 4 | adversarial | Manager có thể dùng VPN cá nhân (NordVPN) không? | 0.417 | faithfulness |
| 5 | adversarial | Thâm niên bao nhiêu năm thì được cộng thêm ngày phép? | 0.417 | faithfulness |
| 6 | adversarial | Nhân viên thử việc có được nghỉ phép năm không? | 0.417 | faithfulness |
| 7 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.500 | faithfulness |
| 8 | multi_hop | Tạm ứng 8 triệu, chưa thanh toán sau 30 ngày (quá hạn)… | 0.500 | faithfulness |
| 9 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.536 | context_recall |
| 10 | multi_hop | Tạm ứng 4 triệu và một nhân viên khác tạm ứng 7 triệu… | 0.614 | faithfulness |

9/10 câu tệ nhất có `worst_metric = faithfulness`. Diagnosis + suggested_fix của từng câu nằm
trong output `bottom_10()` (tra từ `DIAGNOSTIC_TREE`).

---

## 3. Failure Cluster Matrix

*(Chỉ đếm câu **fail thật sự**: điểm của `worst_metric` < `FAILURE_THRESHOLD` = 0.7.
Nếu đếm mọi câu thì mỗi câu luôn có một `worst_metric` và matrix chỉ phản ánh số câu của từng
distribution — không nói lên chỗ hỏng.)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 2 | 11 | 5 | **18** |
| context_recall | 4 | 2 | 3 | 9 |
| context_precision | 0 | 1 | 0 | 1 |
| answer_relevancy | 0 | 1 | 1 | 2 |
| **Total fail** | 6/20 | 15/20 | 9/10 | **30/50** |
| **Tỉ lệ fail** | 30% | 75% | **90%** | 60% |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** `adversarial` — tỉ lệ fail 9/10 = **90%**
**Dominant metric:** `faithfulness` — 18/30 câu fail

**Insight (sinh tự động):** *"30/50 câu fail (worst_metric < 0.7). Distribution 'adversarial' có
tỉ lệ fail cao nhất (9/10 = 90%). Metric 'faithfulness' hỏng nhiều nhất (18 câu). Chẩn đoán: LLM
hallucinating. Fix: Tighten system prompt, lower temperature."*

**Lý do phân tích:**

Corpus có nhiều cặp tài liệu cùng chủ đề khác phiên bản: `nghi_phep_nam_v2023.md` vs
`nghi_phep_nam_v2024.md`, `mat_khau_v1.md` vs `mat_khau_v2.md`. Hai bản gần như trùng từ ngữ nên
retriever kéo cả hai vào top-k (`context_precision` vẫn 0.958 vì cả hai *đều liên quan*), rồi
LLM trộn số của bản cũ với bản mới → `faithfulness` rơi. Câu #2 (đổi mật khẩu) và #5 (thâm niên
cộng ngày phép) là đúng kiểu bẫy này.

`multi_hop` fail 75% vì lý do khác: câu hỏi cần cộng/trừ nhiều nguồn (phụ cấp + thâm niên, tạm
ứng + phí phạt pro-rata). Khi thiếu một mảnh, LLM tự suy diễn con số thay vì nói thiếu dữ liệu —
đúng định nghĩa hallucination mà `faithfulness` đo. 11/15 câu fail của nhóm này rơi vào
faithfulness.

`factual` khoẻ nhất (fail 30%, avg 0.885) và khi fail thì hỏng ở `context_recall` (4/6) chứ không
phải faithfulness — tức là single-doc lookup mà chunk cần thiết không lọt top-k, ví dụ câu #9
(ngưỡng phê duyệt 55 triệu nằm trong bảng phân cấp mà chunk bị cắt).

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness (18 câu) | LLM hallucinating — trộn version, tự suy diễn số liệu | Siết system prompt: "chỉ dùng số có trong context, nêu rõ version, thiếu dữ liệu thì trả lời thiếu"; temperature = 0; bắt trích dẫn tên file nguồn |
| context_recall (9 câu) | Chunk cần thiết không lọt top-k | Tăng `RERANK_TOP_K` từ 3 lên 5, nới parent size cho bảng phân cấp phê duyệt, query decomposition cho multi_hop |
| context_precision (1 câu) | Chunk thừa lọt vào context | Metadata filter theo `effective_date` để loại bản hết hiệu lực trước khi rerank |
| answer_relevancy (2 câu) | Trả lời lệch trọng tâm câu hỏi | Sửa prompt template, tách câu hỏi nhiều ý thành truy vấn con |

Ưu tiên: **prompt + metadata version filter trước, tăng top-k sau**. Vì `context_precision` đã
0.95, tăng top-k chỉ thêm nhiễu chứ không sửa được nguyên nhân chính.

---

## 6. Nhận xét về Adversarial Distribution

`avg_score(adversarial) = 0.649 < avg_score(factual) = 0.885` — **đúng kỳ vọng của bonus Phase A**.
Chênh 0.236 điểm và tỉ lệ fail 90% vs 30% cho thấy bộ 10 câu adversarial thực sự bẫy được
pipeline, tức test set có giá trị chẩn đoán chứ không phải noise.

Pipeline **bị nhầm bởi version conflict**: 4/10 câu adversarial nằm trong bottom 10, trong đó câu
"Bao lâu phải đổi mật khẩu một lần?" (`mat_khau_v1` vs `v2`) và "Thâm niên bao nhiêu năm thì được
cộng thêm ngày phép?" (`nghi_phep_nam_v2023` vs `v2024`) là hai case version conflict kinh điển.
`answer_relevancy` của nhóm này chỉ 0.437 — thấp nhất toàn bộ ma trận — nghĩa là khi gặp bẫy phủ
định ("có được… không?") pipeline còn trả lời lệch cả trọng tâm câu hỏi.

Đòn bẩy cao nhất cho nhóm này không nằm ở prompt mà ở **metadata**: gắn `version` +
`effective_date` cho từng chunk ngay ở M1/M5, lọc cứng bản hết hiệu lực trước khi vào context.
Lúc đó LLM không còn cơ hội chọn nhầm bản cũ.
