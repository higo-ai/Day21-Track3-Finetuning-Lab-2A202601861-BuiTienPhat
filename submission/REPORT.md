# Lab 21 — Evaluation Report

**Họ tên**: Bùi Tiến Phát  **MSSV**: 2A202601861  **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (mặc định) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? (Không cần xử lý thêm vì hệ thống đã tự động bảo lưu khối suy nghĩ `reasoning preserved`).

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
======================================================================
mode = assistant-only   supervised 39/94 (41%)
--- LOSS TÍNH TRÊN ĐOẠN NÀY ---
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3341.4 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1044.8 |
| (c) LoRA fine-tune | 0.970 | 0.4556 | 1.000 | 1450.6 |

**(b) có thật sự mạnh hơn (a) không?** Có.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
(Không sửa đổi, giữ nguyên mẫu gốc để đảm bảo tính khách quan và công bằng cho phép đo).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 0.0001 | 0.6263 | 0.9700 | 1.000 | 12.01 |
| `attn_only` | q,v | 283 | 32456704 | 0.0001 | 0.5369 | 0.9700 | 1.000 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32464896 | 1e-05 | 1.5704 | 0.0000 | 0.000 | 12.01 |
| `qlora` | text-linear | 16 | 32464896 | 0.0001 | 0.7058 | 0.9400 | 1.000 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**

Trên tập target, `attn_only` và `correct` cho kết quả hòa nhau khi cùng đạt điểm số cao tuyệt đối là 0.9700. Tuy nhiên, thứ tự này hoàn toàn không giống với thứ tự theo train loss vì train loss của `attn_only` tốt hơn (0.5369 so với 0.6263 của `correct`). Điểm đặc biệt là `attn_only` có độ trễ sinh từ thấp hơn đáng kể (901.4ms so với 1450.6ms) vì chỉ có 2 module được gắn LoRA thay vì 12, giúp giảm thiểu overhead tính toán khi forward pass. Điều này chứng minh rằng đối với các tác vụ hẹp như trích xuất thông tin, việc nâng rank của LoRA ở các vị trí cốt lõi (`q_proj`, `v_proj`) mang lại hiệu quả tương đương toàn bộ các lớp tuyến tính, đồng thời tối ưu hóa được đáng kể thời gian phản hồi thực tế của mô hình.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

Đường loss của `wrong_lr` (LR 1e-5) giảm rất chậm và phẳng lỳ, bắt đầu từ 2.163 ở bước 5 và chỉ giảm xuống còn 1.119 ở bước 30, trong khi mô hình `correct` (LR 1e-4) giảm cực kỳ nhanh xuống còn 0.0258 ở bước 30. Nếu chỉ nhìn vào biểu đồ loss mà không biết tốc độ học bị đặt sai, chúng ta có thể dễ dàng đi đến kết luận sai lầm rằng mô hình không đủ dung lượng học (rank quá nhỏ) hoặc tập dữ liệu quá phức tạp khiến mô hình bị nghẽn cổ chai. Thử nghiệm này khẳng định rằng LoRA nhạy cảm hơn rất nhiều với tốc độ học so với tinh chỉnh toàn phần (full fine-tuning) và cần một tốc độ học lớn hơn gấp 10 lần (thang 1e-4) để các tham số bổ trợ zero-initialized có thể học được.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

Mô hình QLoRA 4-bit giúp tiết kiệm được 4.92 GB bộ nhớ VRAM đỉnh điểm khi chỉ tiêu tốn 7.09 GB so với 12.01 GB của bản chuẩn `correct` 16-bit. Tuy nhiên, sự đánh đổi này phải trả giá bằng việc độ chính xác trên tập target bị sụt giảm từ 0.9700 xuống còn 0.9400, đồng thời thời gian trễ sinh từ cũng tăng nhẹ từ 1450.6ms lên 1795.1ms do overhead giải nén lượng tử hóa lúc thực thi. Số đo thực tế của dự án hoàn toàn ủng hộ khuyến cáo "không dùng QLoRA cho dòng Qwen3.5" từ nhà phát triển, ngoại trừ trường hợp phần cứng quá hạn chế không thể nạp nổi mô hình 16-bit.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.302` · `valid_trace_rate = 0.00`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)

Cổng hồi quy báo trạng thái FAILED do điểm kiểm tra kiến thức chung ở tập regression bị suy giảm nghiêm trọng tới 0.302 (giảm từ 0.7578 xuống còn 0.4556, vượt xa ngưỡng chịu đựng cho phép là 0.020). Điều này xảy ra do hiện tượng **quên thảm họa (catastrophic forgetting)** khi chúng ta tinh chỉnh mô hình trên một tập dữ liệu quá chuyên biệt (chỉ có 225 mẫu phân loại JSON triage) mà hoàn toàn không pha trộn thêm bất kỳ dữ liệu phổ thông nào. Mô hình bị quá khớp (overfitting) vào cấu trúc JSON và các nhãn phân loại mới, làm mất đi khả năng lập luận chung và kiến thức nền tảng sẵn có của mô hình cơ sở. Để khắc phục điều này cho bài toán thực tế, chúng ta bắt buộc phải trộn thêm từ 1% đến 5% dữ liệu phổ thông (như ShareGPT hoặc các bộ dữ liệu hội thoại thông dụng) vào tập huấn luyện nhằm bảo toàn năng lực lập luận tổng quát của mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô... | van_chuyen / thap / ốp lưng điện thoại / trung_tinh | van_chuyen / trung_binh / ốp lưng điện thoại / trung_tinh | van_chuyen / thap / ốp lưng điện thoại / trung_tinh | ✅ FT thắng: Nhận diện đúng mức độ khẩn cấp thấp ("thap") nhờ học luật phân loại tốt hơn prompt. |
| 2 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | hoi_thong_tin / trung_binh / điện thoại / trung_tinh | hoi_thong_tin / trung_binh / ốp lưng điện thoại / trung_tinh | ✅ FT thắng: Trích xuất đúng tên sản phẩm đầy đủ thay vì bị cắt ngắn thành "điện thoại" như prompt. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien / trung_binh / bình giữ nhiệt / trung_tinh | hoan_tien / trung_binh / bình giữ nhiệt / trung_tinh | hoan_tien / trung_binh / bình giữ nhiệt / tieu_cuc | ❌ **FT thua**: FT đoán sai thái độ thành tiêu cực ("tieu_cuc") trong khi nhãn gốc là trung tính ("trung_tinh"). |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi / trung_binh / nồi chiên không dầu / trung_tinh | san_pham_loi / trung_binh / nồi chiên không dầu / trung_tinh | san_pham_loi / trung_binh / nồi chiên không dầu / tieu_cuc | ❌ **FT thua**: Mô hình quá nhạy cảm với cụm từ "thiếu phụ kiện" và phân loại nhầm thành tiêu cực thay vì trung tính. |
| 5 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm n... | san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh | san_pham_loi / cao / ốp lưng điện thoại / trung_tinh | san_pham_loi / trung_binh / ốp lưng điện thoại / trung_tinh | ✅ FT thắng: Đoán đúng mức khẩn cấp trung bình thay vì bị prompt phóng đại lên thành mức cao. |

Có mẫu chung nào ở các ca FT thua không?

Các ca tinh chỉnh (FT) bị thua chủ yếu gặp lỗi ở phần đánh giá thái độ (sentiment). Mô hình fine-tune có xu hướng bị thiên lệch (bias), tự động gán thái độ tiêu cực ("tieu_cuc") cho các câu chứa từ khóa phàn nàn/yêu cầu hoàn tiền (như "chưa thấy tiền", "thiếu phụ kiện") mặc dù nhãn gốc được coi là trung tính ("trung_tinh"). Điều này cho thấy mô hình bị mất đi sự cân bằng sắc thái tinh tế trong phân tích cảm xúc sau khi học trên tập dữ liệu nhỏ.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?

Chúng ta chưa nên deploy trực tiếp phiên bản fine-tune này lên môi trường sản xuất thực tế. Mặc dù mô hình đạt được sự cải thiện vượt bậc về độ chính xác trên tác vụ đích (tăng từ 76.5% lên 97%), việc suy giảm nghiêm trọng năng lực kiến thức chung ở tập regression (-30.2%) sẽ khiến mô hình dễ gặp lỗi hoặc phản hồi ngớ ngẩn khi người dùng đưa ra các câu hỏi ngoài phạm vi hẹp. Đòn bẩy thực tế lớn nhất trong bài lab này chính là **tốc độ học (Learning Rate)** và **vị trí đặt adapter (text-linear)**. Việc đặt đúng tốc độ học 1e-4 giúp LoRA hội tụ tối ưu, và việc gắn adapter trên toàn bộ các lớp decoder tuyến tính đảm bảo mô hình nắm bắt đầy đủ thông tin biểu diễn. Để có thể deploy an toàn, chúng ta cần bổ sung thêm 2% đến 5% dữ liệu đa mục đích để khắc phục hiện tượng quên thảm họa trước khi đưa mô hình ra thực tế.

**Ba điều tôi học được** (cụ thể, không generic):
1. LoRA cần tốc độ học lớn hơn đáng kể (thường gấp 10 lần) so với full fine-tuning do các ma trận LoRA bổ trợ được khởi tạo bằng 0 và cần cập nhật đủ nhanh để ghi nhận thông tin mới.
2. Việc tối ưu hóa vị trí gắn adapter (`attn_only` so với `text-linear`) đóng vai trò quan trọng trong việc tối ưu hóa hiệu năng phục vụ thực tế (độ trễ giảm 38%), vì số lượng module LoRA ít hơn làm giảm số lượng phép toán cần forward ở thời gian thực.
3. Không được đánh giá mô hình chỉ dựa vào các chỉ số thay thế (như train loss hay validation loss) mà bắt buộc phải kiểm tra thông qua các tác vụ thực tế (downstream tasks) và các bài test cổng hồi quy tổng quát.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ tiến hành pha trộn thêm 2% đến 3% tập dữ liệu ShareGPT phổ thông vào tập dữ liệu huấn luyện để chạy lại và kiểm tra xem mô hình có vượt qua được cổng hồi quy năng lực chung hay không. Đồng thời, tôi sẽ thử nghiệm quét các mức rank nhỏ hơn như r=8 để xem có thể tiếp tục giảm thiểu dung lượng mô hình mà vẫn duy trì độ chính xác cao hay không.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

