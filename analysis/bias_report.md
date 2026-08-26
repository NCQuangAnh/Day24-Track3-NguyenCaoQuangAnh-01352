# LLM Judge Bias Report — Phase B

**Sinh viên:** Nguyễn Cao Quang Anh (01352)
**Ngày:** 2026-08-26
**Judge model:** gpt-4o-mini, `temperature=0`, `response_format=json_object`
**Nguồn số liệu:** `reports/judge_results.json`

---

## 1. Pairwise Judge Results

*(5 cặp answers trong `DEMO_PAIRS`, `src/phase_b_judge.py`)*

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt (judge) |
|---|---|---|---|
| 1 | Nghỉ bao nhiêu ngày phép năm? | A | "cung cấp thông tin chính xác và đầy đủ về số ngày phép theo chính sách hiện hành" |
| 2 | Phụ cấp ăn trưa bao nhiêu? | A | "A cung cấp mức cụ thể, B chỉ nêu chung chung" |
| 3 | Thử việc bao lâu? | A | "A chính xác và cụ thể về thời gian thử việc và mức lương" |
| 4 | Quy trình tạm ứng lương? | A | "A chi tiết quy trình nộp đơn, hạn mức, thời gian duyệt" |
| 5 | Chính sách làm việc từ xa? | A | "A nêu cụ thể số ngày WFH và yêu cầu đăng ký" |

Judge nhất quán ưu tiên answer có **số liệu cụ thể + dẫn version** hơn answer chung chung —
đúng tiêu chí accuracy/completeness đã đặt trong prompt.

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner (đã convert về space gốc) | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | A | A | A | Có |
| 2 | A | A | A | Có |
| 3 | A | A | A | Có |
| 4 | A | A | A | Có |
| 5 | A | A | A | Có |

**Position bias rate:** 0.0% (0/5 case không nhất quán)

gpt-4o-mini ở `temperature=0` giữ nguyên kết luận khi đảo thứ tự A/B trên cả 5 cặp. Lưu ý mẫu chỉ
5 cặp và các cặp đều có chênh lệch chất lượng rõ — position bias thường lộ ra ở các cặp *sát nhau*,
nên con số 0% chưa đủ để kết luận judge miễn nhiễm.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json`
**Judge labels:** `judge_labels_from_human_set()` → chấm **pointwise có reference**: mỗi
`model_answer` được so với `ground_truth` lấy từ `test_set_50q.json` theo `question_id`, judge
trả 1 nếu khớp mọi con số / ngưỡng / cấp phê duyệt / version, 0 nếu sai hoặc thiếu ý bắt buộc.

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Có |
| 5 | 0 | 0 | Có |
| 12 | 1 | 0 | Không |
| 21 | 1 | 1 | Có |
| 23 | 1 | 1 | Có |
| 29 | 0 | 0 | Có |
| 33 | 1 | 1 | Có |
| 41 | 0 | 0 | Có |
| 46 | 1 | 1 | Có |
| 50 | 0 | 0 | Có |

- p_o = 9/10 = 0.90
- Judge: 6 nhãn `1`, 4 nhãn `0`; Human: 6 nhãn `1`, 4 nhãn `0`
- p_e = (0.6 × 0.6) + (0.4 × 0.4) = 0.52
- **Cohen's κ = (0.90 − 0.52) / (1 − 0.52) = 0.792 ≈ 0.800**

**Interpretation:** `almost perfect` (Landis–Koch) — **đạt bonus Phase B (κ > 0.6)**.

Case bất đồng duy nhất là câu 12 ("thưởng Tết tối thiểu 1 tháng lương"): human chấm 1 ("đúng và
súc tích"), judge chấm 0 vì answer thiếu điều kiện "nhân viên chính thức làm việc từ 6 tháng trở
lên" có trong ground truth. Judge **khắt khe hơn human** về tính đầy đủ — sai lệch theo hướng an
toàn cho một quality gate.

**Ghi chú thiết kế quan trọng:** cách chấm ban đầu (so `model_answer` với một "weak answer" né
tránh bằng pairwise) cho κ = **0.000** vì judge luôn chọn `model_answer` → 10/10 nhãn đều là 1,
không phân biệt được gì. Phải chuyển sang **pointwise có ground truth** thì κ mới lên 0.800.
Bài học: judge chỉ tin được khi có reference; pairwise không reference chỉ đo "cái nào trông tốt
hơn", không đo "cái nào đúng".

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie): 5 case decisive.

- A thắng + A dài hơn B: **4 / 5** case
- B thắng + B dài hơn A: **0 / 5** case
- **Verbosity bias rate:** 0.80 (**80%**)

**Kết luận:** judge có xu hướng chọn answer dài hơn. Vấn đề là độ dài không tương quan với độ
đúng — answer dài trích sai version vẫn có thể thắng answer ngắn đúng số liệu. Tuy nhiên trong
lab này con số 80% bị **confound**: các answer A vừa dài hơn *vừa* đúng hơn (chúng chứa số liệu
cụ thể). Muốn đo sạch phải tạo cặp cùng nội dung nhưng khác độ dài (một bản chèn filler) rồi đo
lại; đó là thí nghiệm tiếp theo nên làm trước khi kết luận judge thiên vị độ dài.

---

## 5. Nhận xét chung

κ = 0.800 (`almost perfect`) nên judge **đủ tin cậy để xếp hạng và lọc câu trả lời xấu**, với điều
kiện luôn cấp ground truth. Ngưỡng dùng làm CI gate tôi vẫn đặt ở κ ≥ 0.6 và re-label 20 câu mỗi
tuần để phát hiện drift.

Position bias đo được 0% nên không đáng lo trong setup này, nhưng mẫu 5 cặp là quá nhỏ và các cặp
đều lệch chất lượng rõ. Vẫn nên giữ swap-and-average: chi phí chỉ gấp đôi số LLM call, đổi lại
mọi bất đồng giữa hai pass bị hạ xuống `tie` thay vì thành một kết luận sai — an toàn hơn nhiều
so với tin một pass duy nhất.

Trong production tôi sẽ: (1) giữ `temperature=0` + swap-and-average, (2) luôn dùng pointwise có
reference cho mọi phép đo dùng làm gate, (3) thêm ràng buộc chống verbosity vào prompt — "answer
ngắn mà đúng số liệu và đúng version thắng answer dài chung chung", (4) chỉ cho judge chặn deploy
khi κ trên tập re-label gần nhất còn > 0.6, ngoài ngưỡng đó thì chuyển sang human review.
