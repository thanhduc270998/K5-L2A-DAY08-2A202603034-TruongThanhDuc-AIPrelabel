# Báo cáo Day 08 — <<MSHV - Họ và tên>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | <<YOLO26m>> | <<theo lớp, từ threshold_sweep>> | <<>> |
| three_wheeler | <<YOLOE-26s, prompt: ...>> | <<>> | <<>> |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): <<>>
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: <<chất lượng mask, tốc độ, RAM, quyền truy cập>>
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? <<>>
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: <<>>

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 tiny (`sam2.1_t.pt`) | 3 (2 `+`, 1 `-`) | Có: điểm âm `(720,300)` tách được áo/quần người lái nhưng vẫn dính nhẹ dép/bàn chân và lem lên 3 người ngồi ăn phía sau qua khoang hở; không dính hàng hóa | Lem xuyên qua khoang hở sang người ở nền sau; sót biển `TAXI` trên nóc | `tuktuk_img013_sam2.jpg` |
| `img_013.jpg` | SAM 1 MobileSAM (`mobile_sam.pt`) | 3 (2 `+`, 1 `-`) | Có: tách người ngồi quán phía sau sạch hơn `sam2.1_t`, nhưng vẫn dính vùng má/mặt và bàn chân người lái dù có điểm âm ở lưng | Khuyết một phần đáy lốp sau, sót biển `TAXI` và có lỗ nhỏ ở sàn để chân | `tuktuk_img013_sam1.jpg` |
| `img_004.jpg` | SAM 2.1 tiny (`sam2.1_t.pt`) | 3 (2 `+`, 1 `-`) | Không (xe không có người lái; điểm âm `(465,450)` giúp không dính người đứng cạnh và không dính hàng hóa) | Lem bóng đổ xuống nền gạch đỏ dưới bánh trước; mui bạt bị rỗ/thủng nhiều mảng và sót biển `TAXI` | `tuktuk_img004_sam2.jpg` |
| `img_017.jpg` | SAM 2.1 tiny (`sam2.1_t.pt`) | 3 (2 `+`, 1 `-`) | Không (xe không có người lái; không lem sang hai người đi bộ bên trái hay hàng hóa phía sau) | Điểm âm `(645,355)` đặt trúng lên mui chữ `MR.K` làm khuyết nham nhở mép dưới mui; dính vài đốm nền tối trong khoang xe và sót biển `TAXI` | `tuktuk_img017_sam2.jpg` |

- Đối chiếu guideline mục 4.2:
  - **Quy tắc giúp ích:** Mục 4.2.2 (xử lý lỗ thủng lớn ở khung hở) giải thích vì sao phần nền và người ngồi quán nhìn xuyên qua khoang tuk-tuk (`img_013.jpg`) phải được tách bỏ; mục 4.2.3 (`bóng đổ không thuộc vật`) giúp nhận ra lỗi lem bóng dưới bánh trước ở `img_004.jpg`; mục 4.2.4 (`hai vật chạm nhau không chồng lấn`) khẳng định người lái ở `img_013.jpg` thuộc lớp `person` riêng nên cần thêm click âm ở đầu và chân để tách dứt điểm.
  - **Quy tắc còn thiếu (cần chốt ở v1.1):** Chưa quy định rõ phần nhìn xuyên qua kính chắn gió trong suốt (`img_004.jpg`, `img_017.jpg` — câu hỏi mở 5) thuộc mask xe hay nền, và biển chữ `TAXI` gắn nhô trên nóc xe (cả 4 lần chạy đều bỏ sót) có bắt buộc bao vào mask `three_wheeler` hay không.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Chưa dùng trực tiếp trên CVAT vì CVAT Docker hiện tại chưa cài serverless Nuclio; đã thay thế bằng `scripts/sam2_points.py` offline (cùng cơ chế click dương/âm, nhưng offline phải tự nhập tọa độ pixel và kiểm lại ảnh `.jpg` xuất ra thay vì xem và chỉnh sửa trực tiếp thành polygon trên canvas CVAT).

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
