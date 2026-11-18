# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603034 - Trương Thành Đức

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 2 | Khối xe máy đỗ dày đặc gần camera hay tầm trung: từng xe hay `ignore_region`? | Vật rõ cao `>= 20 px` ở gần/tầm trung bắt buộc gán từng box riêng; chỉ dùng `ignore_region` cho cụm `>= 5` vật ở xa/bị che có chiều cao `< 20 px` không thể tách từng cá thể. | `img_018.jpg`, `img_024.jpg` | `CAL-04` |
| 3 | Hàng nhô ra trên xe đạp / xe ba bánh: box xe và mask có gồm hàng không? | Chỉ tính giỏ/giá gắn cố định liền khung xe; hàng hóa rời nhô ra ngoài khung quá `~20%` chiều dài xe không kéo box theo (mâm/rổ đặt gọn trên giá chính vẫn ôm sát phần giá). | `img_031.jpg`, `img_003.jpg` | `CAL-02` |
| 5 | Kính chắn gió trong suốt và bộ phận gắn cố định trên nóc xe (như biển `TAXI`): thuộc mask xe hay không? | Kính chắn gió khung liền và biển hiệu `TAXI` gắn trên nóc thuộc mask xe; chỉ loại trừ phần nền/người nhìn xuyên qua khoang hở lớn bên hông xe. | `img_004.jpg`, `img_017.jpg` | `CAL-05` |

*(Ghi chú các câu hỏi mở còn lại trong mục 6 v1.0: câu #1 xe đạp điện, câu #4 ngưỡng che 20%, câu #6 ảnh đêm, câu #7 ngưỡng `conf` cố định chưa phát sinh thay đổi nhãn trong 24 ảnh đã duyệt nên giữ nguyên xử lý thận trọng theo v1.0 và đánh dấu `can_review` nếu gặp ở lô tiếp theo.)*

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, lỗi QC)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_002.jpg`, `img_003.jpg` | Tuk-tuk có mui che vs xe máy 2 bánh chở sọt hàng | `YOLO26n` gọi nhầm tuk-tuk thành `truck` (`img_002`); `YOLOE` gán nhầm xe máy chở sọt `#6` thành `three_wheeler` (`img_003`) | Đếm số bánh và kết cấu khung: xe 2 bánh chở sọt gán `motorbike`, chỉ xe 3 bánh thật mới gán `three_wheeler` (`CAL-01`) | Bổ sung vào **Mục 2** (bảng phân biệt `motorbike` vs `three_wheeler`) và **Mục 4.3** |
| `img_035.jpg`, `img_014.jpg` | Box máy bị lệch/lỏng cạnh hoặc kéo rộng sang người ngồi cạnh | `YOLO26n` tạo box `motorbike` `#115` dư `~75 px` trên yên (`img_014`); box `#431`, `#429` (`img_035`) kéo lấn sang người ngồi cạnh và lọt bẫy `loose` | Bắt buộc soi đủ 4 cạnh tiếp xúc ở bước 4 checklist (sai số `<= ~2 px`), không để khoảng trống mặt đường trên yên xe hoặc kéo đè sang người (`CAL-03`) | Bổ sung vào **Mục 4.1.1** (kiểm viền 4 cạnh) |
| `img_014.jpg` | Hai người cùng ngồi trên một xe máy (người lái + người ngồi sau mũ xanh) | `YOLO26n` gộp 2 người vào 1 box `person` `#107` | Tách thành 2 box `person` riêng biệt khi nhìn thấy đủ phần đầu/thân (`>= 20%`) của từng người | Làm rõ tại **Mục 4.1.4** (người ngồi trên xe) |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt quy tắc phân biệt `motorbike` chở sọt vs `three_wheeler`; chốt quy tắc hàng nhô ra, tiêu chí `ignore_region`, chuẩn kiểm viền 4 cạnh box xe máy/tách 2 người trên xe, và phạm vi kính/biển `TAXI` trong mask SAM | `CAL-01` (`img_002`, `img_003`), `CAL-02` (`img_031`), `CAL-03` (`img_035`, `img_014`), `CAL-04` (`img_018`, `img_024`), `CAL-05` (`img_004`, `img_017`) |
