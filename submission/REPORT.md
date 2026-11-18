# Báo cáo Day 08 — 2A202603034 - Trương Thành Đức

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào).

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | `YOLO26n` (`yolo26n.pt`, size `n`) | `car=0.45`; `person`, `bicycle`, `motorbike`, `bus`, `truck` giữ mặc định `0.3` (từ `submission/qc/threshold_sweep.md`) | `r01` (`conf=0.25`) nhận diện tốt `person`, `motorbike`, `car`; khi tăng chung lên `0.5` ở `r04` mất nhiều vật nhỏ/xa (`148 -> 81` box). Khi quét trên 4 ảnh gold: `car` đạt recall `100%` (`>= 90%`) và precision `67%` tại `0.45` (8 vật gold); `person` (max recall `69%`, 26 vật), `bicycle` (`29%`, 7 vật), `motorbike` (`62%`, 13 vật) không ngưỡng `<= 0.5` nào đạt 90% recall nên giữ `0.3`; `bus` (1 vật) và `truck` (0 vật) có `< 5` mẫu nên giữ mặc định `0.3`. |
| three_wheeler | `YOLOE-26s` (`yoloe-26s-seg.pt`, prompt mặc định từ `taxonomy.json`: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.15` (từ `submission/qc/threshold_sweep.md`) | `r02` cho thấy `YOLOE` tìm được lớp `three_wheeler` ngoài 80 lớp COCO (mà `YOLO26n` ở `r01` không có). Tại ngưỡng `0.15` trên 4 ảnh gold (6 vật), `three_wheeler` đạt recall `100%` và precision `67%`. |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "car=0.45,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn**: Chọn **`sam2.1_t.pt` (SAM 2.1 tiny)** làm phiên bản chính và đối chiếu thêm với **`mobile_sam.pt` (SAM 1)**. Lý do: máy chạy PyTorch bản CPU (`2.15.0+cpu`, không dùng CUDA), nên các bản nhẹ (~78 MB và ~41 MB) chạy ổn định, không tốn nhiều RAM như `sam2.1_b.pt` và không cần xin quyền truy cập gated như SAM 3; về chất lượng trên `img_013.jpg`, `sam2.1_t.pt` bám sát viền khung xe tốt nhưng hay lem qua khoang hở ra nền sau, còn `mobile_sam.pt` ít lem nền sau hơn nhưng dễ dính vào mặt/chân người lái và khuyết nhẹ mép lốp.
- Lỗi nào **không sửa được bằng đổi ngưỡng**? `YOLO26n` gán nhầm tuk-tuk thành `truck` (`img_002.jpg` điểm `0.37`, `img_011.jpg` điểm `0.28`) hoặc bỏ sót hoàn toàn xích lô (`img_032.jpg`) vì không có lớp `three_wheeler` trong 80 lớp COCO. Tăng hay giảm `--conf` không đổi được tên lớp; cách sửa là chạy `YOLOE` open-vocabulary cho `three_wheeler` và đặt `yoloe_all` đứng trước `y26_all` trong `merge_prelabels.py` để ưu tiên `three_wheeler` đè lên `car`/`truck` khi `IoU >= 0.6`.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng**:
  - Điểm cao mà sai: trong `pre_traps`, `img_023.jpg` có box `three_wheeler` điểm `0.63` nhưng bao trùm rộng cả tường và mặt đường; `img_029.jpg` có box `bicycle` điểm `0.60` đặt nhầm lên người/xe máy dù không có xe đạp.
  - Điểm thấp mà đúng: trong `t_y26` (`img_011.jpg`), `person` (`0.28`) và `car` (`0.29`) ở tầm xa đều là vật thật (nếu nâng ngưỡng lên `0.5` ở `r04` sẽ bị rơi mất); ở `img_002.jpg`, `motorbike` (`0.27`) góc dưới phải cũng là vật thật.
- Giới hạn: Chọn ngưỡng trên chính `4` ảnh gold (`63` đối tượng trong `submission/labels/gold_manual.json`, `52` đối tượng non-ignore khi đánh giá) rồi đo lại trên chính tập đó (in-sample) nên các con số precision/recall trên gold mang tính **lạc quan**; ngoài ra số mẫu mỗi lớp rất ít (`bus=1`, `truck=0`, `three_wheeler=6`, `bicycle=7`) nên ngưỡng chỉ là ước lượng thô.

## 2. AI error profile (từ `qc/review_stats.md`)

*(Đã loại `28` đối tượng nằm quanh các bẫy khỏi bảng dưới đây; mẫu gồm `24` ảnh batch với `326` box máy và `415` đối tượng sau duyệt, ngưỡng `accepted`: `IoU >= 0.9`.)*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 249 | 249/326 = 76.4% | 326 box máy |
| edited (sửa) | 37 | 37/326 = 11.3% | 326 box máy |
| deleted (xoá) | 39 | 39/326 = 12.0% | 326 box máy |
| added (vẽ thêm) | 128 | 128/415 = 30.8% | 415 đối tượng sau duyệt |

*(Trong `deleted + added` có `6` trường hợp đổi lớp, nên số vật máy sót thật khoảng `128 - 6 = 122` vật.)*

- Lớp nào máy tệ nhất?
  - Tệ nhất về **box thừa/sai (`deleted`)** là **`three_wheeler`** với `23` box bị xóa (chỉ `11` accepted, `7` edited), vì `YOLOE` ở ngưỡng `0.15` sinh nhiều khung trùng/nhầm lên xe máy chở sọt (`img_003`), xe bán tải hoặc nền (`img_002`, `img_008`, `img_023`).
  - Tệ nhất về **bỏ sót (`added`)** là **`person`** (`43`), **`motorbike`** (`42`) và **`car`** (`28`) ở các cảnh đông đúc hoặc bị che khuất một phần (`img_018`, `img_021`, `img_024`, `img_005`); lớp **`bicycle`** cũng đạt rất thấp (chỉ `1` accepted so với `6` added và `2` deleted).
- So với gold: Precision của **máy** là `37/43 = 86.0% (KTC 95%: 73–93%)`, Recall của **máy** là `37/52 = 71.2% (KTC 95%: 58–82%)` (`TP=37, FP=6, FN=15`; mẫu số recall: `52` đối tượng gold ở `4` ảnh; số đo này lạc quan do chọn ngưỡng in-sample trên chính gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: Ở **vòng duyệt 1**, chỉ bắt được **`8/12 = 66.7%`** (`KTC 95%: 39–86%`), dưới ngưỡng `80%`, để lọt `4` bẫy: `img_035.jpg (loose)`, `img_025.jpg (loose)`, `img_004.jpg (missing)` và `img_021.jpg (extra)` (trong đó `loose` chỉ đạt `1/3`, `missing` `2/3`, `extra` `2/3`). Sau khi phát hiện dưới ngưỡng, mình đã **rà duyệt lại toàn lô lên vòng 2** và đạt **`11/12 = 91.7%`** (`KTC 95%: 65–99%`), chỉ còn lọt `1` bẫy **`img_035.jpg (loose)`** (`wrong_class: 3/3`, `loose: 2/3`, `missing: 3/3`, `extra: 3/3`).
  - Nguyên nhân lọt: ở vòng 1 đã bỏ qua bước 0 (ẩn hết nhãn máy để tự đếm vật trước nên lọt `missing`/`extra`) và bỏ sót bước 4 soi 4 cạnh viền (nên lọt `2/3` bẫy `loose`). Ở vòng 2 đã sửa được `missing`/`extra`, nhưng vẫn lọt `1` bẫy `loose` trên `img_035.jpg` do lệch viền không lớn trên nền xe máy đông nên mắt vẫn bị neo vào khung máy.
- `accepted` cao (`77.3%` vòng 1 / `76.4%` vòng 2) nhưng bắt bẫy vòng 1 thấp (`66.7%`) nói lên điều gì? Nó chứng tỏ chỉ số `accepted` cao **không** đồng nghĩa nhãn đã sạch: người duyệt bị **thiên kiến tự động hóa (automation bias)**, thấy khung của máy "trông hợp lý" nên lướt qua nhanh mà không kiểm tra lại bằng mắt ở biên box hoặc khoảng trống xung quanh.

## 4. Máy có đáng dùng không?

- Thời gian (từ `submission/time_log.csv`): gán tay `25 phút / 4 ảnh = 6.25 phút/ảnh` (`img_001: 7p, img_012: 6p, img_020: 4p, img_036: 8p`) vs duyệt có pre-label `67 phút / 24 ảnh = 2.79 phút/ảnh` (từ `1` phút ở ảnh thưa đến `5` phút ở ảnh dày như `img_018, img_021, img_028, img_029`).
  - Thiên lệch cần nêu: 4 ảnh gold được gán tay trước nên khi sang `batch_review` đã quen thao tác CVAT và bối cảnh phố Đông Nam Á; đồng thời thời gian `batch_review` có bao gồm công rà lại vòng 2 cho đạt chuẩn bắt bẫy.
- Kết luận: **Có đáng dùng (hiệu quả cho các lớp phổ biến `person`, `motorbike`, `car`)**, vì giúp giảm thời gian trung bình xuống còn `2.79` so với `6.25 phút/ảnh` (nhanh hơn ~`2.24x`) với `249/326 = 76.4%` box máy giữ nguyên được; tuy nhiên máy vẫn sót `128/415 = 30.8%` đối tượng (`122` sót thật) và lớp `three_wheeler` của `YOLOE` có `23` box bị xóa nên bắt buộc phải qua khâu review chặt chứ không thể dùng trực tiếp.
- **Ba quy tắc giảm thiên kiến của bạn**:
  1. Ở **100% ảnh**, ẩn toàn bộ nhãn máy (biểu tượng con mắt trong bảng *Objects*) trong ít nhất `15 giây` đầu để tự đếm nhẩm người/xe trước khi hiện lại box máy (kiểm bằng việc bắt đủ `3/3` bẫy `missing` và `3/3` bẫy `extra`).
  2. Zoom kiểm tra **đủ 4 cạnh tiếp xúc** (`sai số <= ~2 px`) cho mọi box máy có sẵn kể cả box có `conf >= 0.7`, đảm bảo không giữ box lệch/đè người (kiểm bằng tỷ lệ bắt bẫy `loose >= 2/3` và không lặp lỗi `sai_vi_tri_kich_thuoc` khi kiểm QC).
  3. Nghỉ giải lao ngắn `3–5 phút` sau mỗi `8–10` ảnh (ghi dòng `break` vào `time_log.csv`), đặc biệt không duyệt liên tục quá 2 ảnh dày xe máy (`img_018, img_024, img_028, img_029`) liên tiếp.

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

- Số quyết định: `5` quyết định trong `submission/qc/calibration_decisions.csv` (`CAL-01` đến `CAL-05`); quan trọng nhất là **`CAL-03`** (chuẩn hóa kiểm tra 4 cạnh tiếp xúc box máy `<= ~2 px`, không kéo lấn sang người ngồi cạnh hoặc để khoảng trống trên yên xe như `img_035 #431/#429` và `img_014 #115`) vì lỗi `sai_vi_tri_kich_thuoc` chiếm tới `7/9` lỗi trong mẫu QC, cùng **`CAL-01`** (phân biệt xe máy 2 bánh chở sọt hàng `motorbike` với `three_wheeler` trên `img_003 #6`).
- Quy tắc mới đã vào `submission/guidelines/guideline_v1.1_calibrated.md`: chốt 3 câu hỏi mở từ mục 6 của v1.0 (câu #2 tiêu chí `ignore_region`, câu #3 hàng nhô trên xe đạp/ba gác, câu #5 kính chắn gió & biển `TAXI` trong mask) và bổ sung 3 ca khó mới (phân biệt xe máy chở sọt vs `three_wheeler`, kiểm viền 4 cạnh box xe máy, tách 2 `person` trên cùng xe máy ở `img_014.jpg`), mỗi ca đều gắn ảnh ví dụ `img_0xx` và cập nhật bảng lịch sử `v1.0 -> v1.1`.
- Nguyên nhân gốc: Kết hợp cả ba nguyên nhân — (1) **guideline v1.0 còn mở** ở câu hỏi #2, #3, #5 gây lưỡng lự ở ca xe đạp chở mâm trái cây (`img_031 #386`) và vùng đông người; (2) **lỗi máy** từ `YOLOE` ngưỡng thấp và `YOLO26n` gộp 2 người trên xe (`img_014 #107`); (3) **lỗi người duyệt** bỏ sót bước soi 4 cạnh viền (`img_003 #1`, `img_014 #115`, `img_035 #431/#429`) và bỏ sót 2 xe nhỏ/xa ở `img_006`.

## 7. QC và quyết định (Part 6)

- Mẫu: `5` ảnh (`0` ảnh `can_review` vì ở bước review chưa gắn cờ `can_review=true` + `5` ảnh ngẫu nhiên với `seed 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), tổng `79` đối tượng non-ignore (nguồn: `submission/qc/qc_sample.csv` và `submission/qc/qc_report_auto.md`). Người kiểm: tự kiểm trên ảnh overlay `work/qc/review/` kết hợp rà soát hỗ trợ của AI (**hạn chế**: tự kiểm trên lô do chính mình duyệt dễ giữ lại điểm mù quen tay, và mẫu chưa có ca khó `can_review`).
- Lỗi theo **đối tượng**: `9/79 = 11.4%` · theo **ảnh**: `5/5 = 100.0%` (ngưỡng Release `<= 2%`).
- Top lỗi (từ `submission/qc/qc_review.csv` và `qc_report_auto.md`):
  - `sai_vi_tri_kich_thuoc` (`7/9` lỗi): ví dụ `img_003.jpg` (`#1` `three_wheeler` dư `~45 px` phía trên mái bạt), `img_014.jpg` (`#115` `motorbike` dư `~75 px` trên yên xe; `#107` gộp 2 người vào 1 box `person`), `img_031.jpg` (`#386` `bicycle` lệch phải `~50 px` và cắt mâm trái cây sau; `#385` `person` thiếu phần chân dưới `y=438`), `img_035.jpg` (`#431`, `#429` mép trái box `motorbike` kéo lấn sang người `#440`, `#432`).
  - `sai_lop` (`1/9` lỗi): `img_003.jpg` (`#6` xe máy 2 bánh chở sọt hàng bị gán nhãn `three_wheeler`).
  - `bo_sot` (`1/9` lỗi): `img_006.jpg` (bỏ sót ô tô trắng ở cuối ngõ phía sau `#55/#67` và xe máy đỗ sau `#52/#65`).
- **Giới hạn mẫu nhỏ** (từ `submission/qc/qc_report_auto.md`): Với `5/5` ảnh lỗi trong mẫu `5` ảnh, **cận trên 95% (Wilson)** của tỉ lệ ảnh lỗi thật lên tới **`100%`**; ngay cả khi mẫu nhỏ `5` ảnh đạt `0` lỗi thì cận trên Wilson 95% vẫn quá rộng để chứng minh tỉ lệ lỗi thực sự `<= 2%`, nên không thể dùng mẫu nhỏ làm bằng chứng đạt chuẩn nếu chưa sửa lỗi lặp.
- Script kiểm được / không kiểm được: Các script (`validate_labels.py`, `review_stats.py`, `qc_report.py`) kiểm được cấu trúc COCO, tên lớp hợp lệ trong taxonomy, box nằm trong biên ảnh, số đo trên phiếu `qc_review.csv` và tỷ lệ bắt bẫy; **không** kiểm được box có ôm sát vật thật `<= 2 px` hay không, lớp có đúng thực tế không, vật mà cả người và máy cùng bỏ sót ngoài gold, hay việc người duyệt có thực sự ẩn nhãn quét ảnh trước khi sửa không.

**Quyết định: REWORK**, ngày `2026-11-18`. Lý do (đi theo cây `Hold → Rework → Release`): Không chọn `Hold` vì dữ liệu ảnh rõ dùng được (`--data-problem 0`) và các ca lưỡng lự chính của v1.0 đã được chốt sang v1.1 (`--guideline-gap 0`). Xét sang `Rework`: tỉ lệ ảnh lỗi trong mẫu QC là **`5/5 = 100.0% > 2%`** và xuất hiện **lỗi hệ thống `'sai_vi_tri_kich_thuoc'` (`7/9` lỗi, chiếm `>= 50%` và `>= 3` ca)**, nên dù tỷ lệ bắt bẫy đạt `11/12 = 91.7% >= 80%`, lô vẫn không đạt tiêu chuẩn để Release và bắt buộc phải **`REWORK`**.
Việc cần làm tiếp: Mở lại Task `batch-review` trên CVAT, sửa triệt để toàn bộ 24 ảnh theo quy tắc viền mới trong `v1.1` (sửa ngay các lỗi đã chỉ ra ở `img_003 #1/#6`, `img_006`, `img_014 #115/#107`, `img_031 #385/#386`, `img_035 #429/#431` và tick `can_review` cho các ca biên thực sự), xuất các ảnh đã sửa thành `submission/labels/qc_fixed.json`, rồi chạy lại `build_release.py` → `qc_sample.py --force` → `qc_report.py` để nghiệm thu lại vòng sau.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Sử dụng AI Gemini để nhờ AI hướng dẫn cách sử dụng model Auto Anontation, cách cài đặt vào dự án để làm bài tập, bản thân có chủ động đưa số liệu thực tế (Không nhờ AI lấy dữ liệu hay làm hộ bài) để nhờ AI trau chuốt câu từ để viết vào báo cáo cho chỉnh chu
- **Gold có bị nhiễm không**: Không — 4 ảnh gold được gán tay hoàn toàn trong Task `gold-manual` trên CVAT và export thành `submission/labels/gold_manual.json` trước khi chạy model hay xem bất kỳ pre-label nào.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt**: Windows 10 Home Single Language (64-bit); phần cứng có `NVIDIA GeForce MX150` (2 GB VRAM) + `Intel(R) UHD Graphics 620`, nhưng `.venv` cài PyTorch CPU (`2.15.0+cpu`, `cuda_available = False`) nên toàn bộ model chạy trên **CPU** với bộ trọng số nhẹ (`yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`, `mobile_sam.pt`); CVAT Docker local bản `v2.74.1` chạy tại `http://localhost:8080` (chưa cài serverless Nuclio). Vướng mắc cài đặt: ban đầu tải model vào `E:\vinuni\day08\models` nên phải copy sang `models/` của repo rồi mới chạy được model.
- **Dùng cache `data/cache/` thay vì tự chạy?** Không — toàn bộ lượt thử `pilot`, chạy `--split all` và `sam2_points.py` đều tự chạy trực tiếp trên máy.
- **Lỗi / bug script đã gặp**: Không gặp bug từ mã nguồn script; chỉ gặp lỗi thao tác trên PowerShell (chưa kích hoạt `.venv` khi chạy `sam2_points.py`, để placeholder `--thr`, gõ dính chữ vào đường dẫn `--gold`) và phải rà duyệt lại CVAT vòng 2 sau khi vòng 1 chỉ đạt `8/12 = 66.7%` bắt bẫy.
- **Một điều bạn làm khác lần sau**: Tick cờ `can_review` ngay trong lúc gán gold và duyệt lô cho các ca lưỡng lự để mẫu QC tự động rút được đúng các ca khó, đồng thời ẩn nhãn máy quét ảnh trước nghiêm ngặt ngay từ vòng đầu để không phải duyệt lại vòng 2.
