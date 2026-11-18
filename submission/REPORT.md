# Báo cáo Day 08 — 2A202603034 - Trương Thành Đức

> Mọi số liệu đều kèm mẫu số và dẫn theo file trong `submission/qc/`, `submission/time_log.csv`, `submission/model_log.csv`.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| `person`, `bicycle`, `motorbike`, `car`, `bus`, `truck` | `YOLO26n` (`yolo26n.pt`, size `n`) | `car=0.45`; 5 lớp còn lại giữ `0.3` (`threshold_sweep.md`) | `r01` (`conf=0.25`) bắt ổn `person`, `motorbike`, `car`; khi tăng chung lên `0.5` (`r04`) rơi mất gần nửa số box (`148 -> 81`). Quét trên 4 ảnh gold: `car` đạt recall `100%` (precision `67%`, 8 vật) ở `0.45`; `person` (max recall `69%`, 26 vật), `bicycle` (`29%`, 7 vật), `motorbike` (`62%`, 13 vật) không mức `<= 0.5` nào đạt `90%` nên giữ `0.3`; `bus` (1 vật) và `truck` (0 vật) ít hơn 5 mẫu nên giữ mặc định `0.3`. |
| `three_wheeler` | `YOLOE-26s` (`yoloe-26s-seg.pt`, prompt từ `taxonomy.json`: `tuk-tuk`, `cyclo`, `cargo tricycle`) | `0.15` (`threshold_sweep.md`) | `r02` chứng minh `YOLOE` bắt được xe ba bánh mà `YOLO26n` không có trong 80 lớp COCO. Trên 4 ảnh gold (6 vật), ngưỡng `0.15` đạt recall `100%`, precision `67%`. |

- Lệnh `merge_prelabels.py` đã dùng:
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "car=0.45,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM đã chọn:** Chọn **`sam2.1_t.pt` (SAM 2.1 tiny)** là chính, có chạy thêm **`mobile_sam.pt` (SAM 1)** trên `img_013.jpg` để so sánh. Lý do: máy chạy PyTorch CPU nên ưu tiên bản nhẹ (~78 MB và ~41 MB) cho ổn định RAM và không phải xin quyền gated như SAM 3. Khi test trên `img_013.jpg`, `sam2.1_t` bám viền xe tốt nhưng hay lem qua khoang hở ra người ngồi sau; `mobile_sam` ít lem nền sau hơn nhưng lại dính mặt/chân tài xế và hơi khuyết đáy lốp.
- **Lỗi không sửa được bằng đổi ngưỡng:** `YOLO26n` gọi nhầm tuk-tuk thành `truck` (`img_002.jpg` điểm `0.37`, `img_011.jpg` điểm `0.28`) và bỏ sót xích lô (`img_032.jpg`) vì không có lớp `three_wheeler`. Đổi `--conf` không sửa được nhãn lớp; phải dùng `YOLOE` cho `three_wheeler` và đặt trước `y26_all` khi gộp (`IoU >= 0.6`) để đè các box `car`/`truck` trùng chỗ.
- **Box điểm cao nhưng sai / điểm thấp nhưng đúng:**
  - Cao mà sai: trong `pre_traps`, `img_023.jpg` có `three_wheeler` điểm `0.63` trùm cả tường và mặt đường; `img_029.jpg` có `bicycle` điểm `0.60` đè lên xe máy dù không có xe đạp.
  - Thấp mà đúng: trong `t_y26`, `img_011.jpg` có `person` (`0.28`) và `car` (`0.29`) ở xa đều là xe/người thật (bị mất nếu nâng lên `0.5`); `img_002.jpg` có `motorbike` (`0.27`) góc dưới phải cũng đúng.
- **Giới hạn chọn ngưỡng:** Ngưỡng được quét và đo trên cùng `4` ảnh gold (`52` vật non-ignore, trong tổng `63` đối tượng `gold_manual.json`) nên kết quả trên gold bị lạc quan. Ngoài ra các lớp ít mẫu (`bus=1`, `truck=0`, `three_wheeler=6`, `bicycle=7`) khiến ngưỡng ước lượng còn thô.

## 2. AI error profile (từ `qc/review_stats.md`)

*(Tính trên `24` ảnh batch: `326` box máy, `415` đối tượng sau duyệt; đã loại `28` đối tượng quanh bẫy, ngưỡng `accepted`: `IoU >= 0.9`.)*

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 249 | 249/326 = 76.4% | 326 box máy |
| edited (sửa) | 37 | 37/326 = 11.3% | 326 box máy |
| deleted (xoá) | 39 | 39/326 = 12.0% | 326 box máy |
| added (vẽ thêm) | 128 | 128/415 = 30.8% | 415 đối tượng sau duyệt |

*(Trong `deleted + added` có `6` ca đổi lớp, nên máy sót thật khoảng `128 - 6 = 122` vật.)*

- **Lớp máy tệ nhất:**
  - Về box thừa/sai (`deleted`): **`three_wheeler`** (`23` box bị xóa, chỉ `11` accepted và `7` edited) do `YOLOE` ngưỡng `0.15` bắt nhầm xe máy chở sọt (`img_003`), xe bán tải hoặc nền (`img_002`, `img_008`, `img_023`).
  - Về bỏ sót (`added`): nhiều nhất ở **`person`** (`43`), **`motorbike`** (`42`) và **`car`** (`28`) trong các ảnh đông xe/bị che (`img_018`, `img_021`, `img_024`, `img_005`); lớp **`bicycle`** cũng kém (chỉ `1` accepted, sót `6`, xóa `2`).
- **So với gold (`4` ảnh, `52` đối tượng):** Precision của máy = `37/43 = 86.0% (KTC 95%: 73–93%)`, Recall = `37/52 = 71.2% (KTC 95%: 58–82%)` (`TP=37, FP=6, FN=15`; con số lạc quan do chọn ngưỡng in-sample trên gold).

## 3. Thiên kiến tự động hóa và ảnh bẫy

- **Bắt bẫy:** Vòng 1 chỉ bắt được **`8/12 = 66.7%`** (`KTC 95%: 39–86%`), dưới ngưỡng `80%`, lọt `4` bẫy: `img_035.jpg (loose)`, `img_025.jpg (loose)`, `img_004.jpg (missing)`, `img_021.jpg (extra)`. Thấy chưa đạt nên mình quay lại duyệt kỹ toàn bộ 24 ảnh ở vòng 2, nâng lên **`11/12 = 91.7%`** (`KTC 95%: 65–99%`), chỉ còn lọt `1` bẫy **`img_035.jpg (loose)`** (`wrong_class: 3/3`, `loose: 2/3`, `missing: 3/3`, `extra: 3/3`).
  - Lý do lọt: vòng 1 mình không ẩn nhãn máy để đếm vật trước (nên lọt `missing`/`extra`) và lướt qua bước kiểm 4 cạnh viền (nên `loose` chỉ bắt `1/3`). Vòng 2 sửa hết `missing`/`extra` nhưng vẫn sót 1 bẫy `loose` trên `img_035.jpg` do box máy lệch không nhiều trong cảnh đông xe.
- **`accepted` cao nhưng bắt bẫy vòng 1 thấp:** Dù `accepted` đạt ~76–77%, bắt bẫy vòng 1 chỉ `66.7%` cho thấy mình đã tin theo khung sẵn của máy (automation bias): thấy box trông hợp lý nên giữ luôn mà không tự quét ảnh hay soi kỹ 4 cạnh.

## 4. Máy có đáng dùng không?

- **Thời gian (từ `submission/time_log.csv`):**
  - Gán tay 4 ảnh gold: `25 phút / 4 ảnh = 6.25 phút/ảnh` (`img_001: 7p, img_012: 6p, img_020: 4p, img_036: 8p`).
  - Duyệt 24 ảnh batch có pre-label: `67 phút / 24 ảnh = 2.79 phút/ảnh` (từ `1p` ở ảnh ít xe đến `5p` ở ảnh dày như `img_018, img_021, img_028, img_029`).
  - Thiên lệch: mình gán gold trước nên khi vào batch đã quen tay CVAT và quen kiểu đường phố Đông Nam Á; ngoài ra `67` phút đã gồm cả thời gian rà lại vòng 2.
- **Kết luận:** **Đáng dùng cho các lớp phổ biến của YOLO26n (`person`, `motorbike`, `car`)** vì rút thời gian làm xuống hơn một nửa (`2.79` vs `6.25 phút/ảnh`, nhanh hơn ~`2.24x`) và giữ được `249/326 = 76.4%` box máy. Tuy nhiên vẫn bắt buộc phải duyệt kỹ vì máy sót tới `128/415 = 30.8%` đối tượng (`122` sót thật) và `three_wheeler` của `YOLOE` sinh nhiều box rác (`23` deleted).
- **Ba quy tắc giảm thiên kiến của mình:**
  1. Ở **100% ảnh**, bấm ẩn hết nhãn máy ít nhất `15 giây` đầu để tự đếm người/xe trước rồi mới bật lại box (kiểm bằng kết quả bắt đủ `3/3` `missing` và `3/3` `extra`).
  2. Zoom kiểm đủ **4 cạnh tiếp xúc** (lệch `<= ~2 px`) cho mọi box máy kể cả box `conf >= 0.7` (kiểm bằng việc giữ bắt bẫy `loose >= 2/3` và tránh lặp lỗi `sai_vi_tri_kich_thuoc` khi QC).
  3. Nghỉ `3–5 phút` sau mỗi `8–10` ảnh (ghi dòng `break` trong `time_log.csv`), không duyệt liên tiếp các ảnh đông xe (`img_018, img_024, img_028, img_029`).

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | `sam2.1_t.pt` (SAM 2.1 tiny) | 3 (2 `+`, 1 `-`) | Có: click âm `(720,300)` tách được áo/quần lái xe nhưng vẫn dính nhẹ dép và lem sang 3 người ngồi sau qua khoang hở; không dính hàng | Lem qua khoang hở lên người phía sau; sót biển `TAXI` trên nóc | `tuktuk_img013_sam2.jpg` |
| `img_013.jpg` | `mobile_sam.pt` (SAM 1) | 3 (2 `+`, 1 `-`) | Có: ít lem người ngồi quán phía sau hơn `sam2.1_t`, nhưng lại dính mặt và bàn chân người lái | Khuyết nhẹ đáy lốp sau, sót biển `TAXI` và rỗ nhỏ ở sàn xe | `tuktuk_img013_sam1.jpg` |
| `img_004.jpg` | `sam2.1_t.pt` (SAM 2.1 tiny) | 3 (2 `+`, 1 `-`) | Không có lái xe; click âm `(465,450)` tách sạch người áo đen bên cạnh, không dính hàng | Lem bóng đổ xuống gạch đỏ dưới bánh trước; mui bạt bị rỗ và sót biển `TAXI` | `tuktuk_img004_sam2.jpg` |
| `img_017.jpg` | `sam2.1_t.pt` (SAM 2.1 tiny) | 3 (2 `+`, 1 `-`) | Không có lái xe; tách sạch 2 người đi bộ bên trái và đồ trong quán | Click âm `(645,355)` đặt trúng mui chữ `MR.K` làm mép dưới mui bị khuyết nham nhở; còn vài đốm nền tối trong khoang xe và sót biển `TAXI` | `tuktuk_img017_sam2.jpg` |

- **Đối chiếu guideline mục 4.2:**
  - Quy tắc giúp ích: mục 4.2.2 (khoét bỏ lỗ thủng lớn ở khung hở) dùng để loại nền quán/người ngồi sau trên `img_013.jpg`; mục 4.2.3 (bóng đổ không thuộc vật) giúp nhận ra lỗi lem bóng bánh trước ở `img_004.jpg`; mục 4.2.4 (hai vật chạm nhau) cho thấy cần thêm click âm ở đầu/chân để tách hẳn người lái khỏi xe ở `img_013.jpg`.
  - Quy tắc còn thiếu: v1.0 chưa nói rõ phần nhìn xuyên qua kính chắn gió trong suốt (`img_004.jpg`, `img_017.jpg`) có tính vào mask xe không và biển chữ `TAXI` trên nóc xe (cả 4 lần chạy đều sót) có bắt buộc gom vào mask `three_wheeler` không.
- *(Tuỳ chọn)* **Interactor SAM trong CVAT:** Chưa dùng trực tiếp trên CVAT vì Docker mặc định không cài Nuclio serverless; mình chạy `scripts/sam2_points.py` offline để thử cơ chế click dương/âm (bất tiện hơn CVAT ở chỗ phải tự dò tọa độ pixel rồi mở ảnh `.jpg` kiểm tra lại chứ không chỉnh thẳng polygon trên canvas).

## 6. Calibration (Part 6)

- **Số quyết định:** `5` dòng trong `submission/qc/calibration_decisions.csv` (`CAL-01` → `CAL-05`). Quan trọng nhất là **`CAL-03`** (siết kiểm tra 4 cạnh box xe máy/lỏng viền, do lỗi `sai_vi_tri_kich_thuoc` chiếm `7/9` ca lỗi QC) và **`CAL-01`** (phân biệt xe máy 2 bánh chở sọt `motorbike` với `three_wheeler` ở `img_003 #6`).
- **Quy tắc mới đã cập nhật vào `submission/guidelines/guideline_v1.1_calibrated.md`:** Chốt 3 câu hỏi mở của v1.0 (câu #2 tiêu chí `ignore_region`, câu #3 hàng nhô trên xe đạp/ba gác, câu #5 kính chắn gió & biển `TAXI` trong mask SAM) và thêm 3 ca khó mới (xe máy chở sọt vs `three_wheeler`, kiểm viền 4 cạnh box xe máy, tách 2 `person` trên cùng một xe máy ở `img_014.jpg`), tất cả đều có `img_0xx` minh họa và ghi vào lịch sử `v1.0 -> v1.1`.
- **Nguyên nhân gốc:** Do cả 3 phía — (1) guideline v1.0 còn mở ở câu #2, #3, #5 (`img_031 #386`, `img_018`); (2) lỗi model (`YOLOE` gọi nhầm `img_003 #6`, `YOLO26n` gộp 2 người ở `img_014 #107`); (3) mình còn sót bước kiểm viền 4 cạnh ở `img_003 #1`, `img_014 #115`, `img_035 #431/#429` và sót xe nhỏ ở `img_006`.

## 7. QC và quyết định (Part 6)

- **Mẫu kiểm:** `5` ảnh (`0` ảnh `can_review` vì lúc review quên tick + `5` ảnh ngẫu nhiên `seed 42`: `img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`), tổng `79` đối tượng non-ignore (`qc_sample.csv`, `qc_report_auto.md`). Người kiểm: mình tự kiểm trên ảnh overlay `work/qc/review/` có hỗ trợ đối chiếu từ AI (**hạn chế:** tự kiểm bài của mình dễ lặp lại điểm mù và mẫu không có ca khó `can_review`).
- **Tỉ lệ lỗi:** theo **đối tượng** `9/79 = 11.4%` · theo **ảnh** `5/5 = 100.0%` (ngưỡng Release `<= 2%`).
- **Top lỗi (từ `qc_review.csv` và `qc_report_auto.md`):**
  - `sai_vi_tri_kich_thuoc` (`7/9` lỗi): `img_003.jpg` (`#1` dư ~45 px trên mái bạt), `img_014.jpg` (`#115` dư ~75 px trên yên xe; `#107` gộp 2 người), `img_031.jpg` (`#386` lệch phải ~50 px và cắt mâm trái cây; `#385` thiếu chân dưới hông), `img_035.jpg` (`#431`, `#429` lấn sang người `#440`, `#432`).
  - `sai_lop` (`1/9` lỗi): `img_003.jpg` (`#6` xe máy 2 bánh chở sọt bị gán `three_wheeler`).
  - `bo_sot` (`1/9` lỗi): `img_006.jpg` (sót ô tô trắng cuối ngõ sau `#55/#67` và xe máy đỗ sau `#52/#65`).
- **Giới hạn mẫu nhỏ:** Theo `qc_report_auto.md`, với `5/5` ảnh lỗi trên mẫu `5` ảnh, **cận trên 95% (Wilson)** của tỉ lệ ảnh lỗi thật lên tới **`100%`**. Mẫu chỉ có 5 ảnh nên kể cả khi ra `0` lỗi cũng không đủ chứng minh lô đạt `<= 2%`.
- **Script kiểm được / không kiểm được:** Script kiểm được định dạng COCO, tên lớp hợp lệ, box nằm trong ảnh, số đo trên phiếu `qc_review.csv` và bắt bẫy; **không** kiểm được box có ôm sát vật `<= 2 px` chưa, gán đúng lớp thực tế chưa, vật mà cả người và máy cùng sót ngoài gold, hay người duyệt có thực sự ẩn box quét ảnh trước không.

**Quyết định: REWORK**, ngày `2026-11-18`.
- **Lý do (theo cây `Hold → Rework → Release`):** Không `Hold` vì ảnh gốc rõ (`--data-problem 0`) và các ca lưỡng lự đã được chốt sang `v1.1` (`--guideline-gap 0`). Xét `Rework` thấy ảnh có lỗi đạt **`5/5 = 100.0% > 2%`** và có **lỗi hệ thống `sai_vi_tri_kich_thuoc` (`7/9` lỗi)**, nên dù bắt bẫy đạt `11/12 = 91.7% >= 80%`, lô vẫn phải **`REWORK`**.
- **Việc cần làm tiếp:** Mở lại Task `batch-review` trên CVAT, sửa lại toàn bộ lô theo `v1.1` (sửa các lỗi ở `img_003 #1/#6`, `img_006`, `img_014 #115/#107`, `img_031 #385/#386`, `img_035 #429/#431` và tick `can_review` cho ca biên), xuất file sửa thành `submission/labels/qc_fixed.json` rồi chạy lại `build_release.py` → `qc_sample.py --force` → `qc_report.py`.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài:** Có dùng trợ lý AI (Gemini / Antigravity) để hướng dẫn cài đặt môi trường `.venv`, xử lý lỗi nạp Task CVAT, kiểm tra chỉ xem hình học trên `batch_reviewed.json` & Job #21, gợi ý rà ảnh overlay `work/qc/review/` điền `qc_review.csv` và chỉnh câu chữ báo cáo cho gọn rõ. Mình tự thao tác gán gold, duyệt 24 ảnh trên CVAT, chạy toàn bộ các script và mở ảnh/file kiểm tra lại từng số liệu trước khi chốt.
- **Gold có bị nhiễm không:** Không — 4 ảnh gold được gán tay hoàn toàn trong Task `gold-manual` trên CVAT và export `gold_manual.json` trước khi chạy model hay xem pre-label.
- **Máy / GPU / CVAT dùng; vướng mắc cài đặt:** Windows 10 Home 64-bit; máy có `NVIDIA GeForce MX150` (2 GB VRAM) + `Intel(R) UHD Graphics 620`, nhưng `.venv` dùng PyTorch CPU (`2.15.0+cpu`, `cuda_available = False`) nên toàn bộ model chạy trên **CPU** với bộ nhẹ (`yolo26n.pt`, `yoloe-26s-seg.pt`, `sam2.1_t.pt`, `mobile_sam.pt`); CVAT Docker `v2.74.1` chạy tại `http://localhost:8080` (không có Nuclio). Vướng mắc: lúc đầu tải model sang `E:\vinuni\day08\models` nên phải copy vào `models/` của repo, đặt tên `gold-manual.json` bị lệch dấu gạch ngang, và tạo Task #20 quên chọn Project nên bị lỗi thiếu nhãn `motorbike`.
- **Dùng cache `data/cache/` thay vì tự chạy?** Không — tự chạy toàn bộ các lệnh `prelabel.py` và `sam2_points.py` trên máy.
- **Lỗi / bug script đã gặp:** Không có bug trong script; chỉ gặp lỗi thao tác PowerShell (quên kích hoạt `.venv` khi chạy `sam2_points.py`, để placeholder `--thr`, dán dính chữ vào `--gold`) và vòng 1 bắt bẫy đạt `8/12` nên phải duyệt lại vòng 2 lên `11/12`.
- **Một điều làm khác lần sau:** Tick `can_review` ngay lúc gán gold và review để `qc_sample.py` rút đúng các ca biên vào mẫu QC, đồng thời ẩn box máy quét ảnh trước ngay từ vòng 1 để khỏi mất công duyệt lại.
