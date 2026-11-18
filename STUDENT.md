# Thông tin học viên

- Họ và tên: Trương Thành Đức
- MSHV: 2A202603034
- Lớp (L2 / L3 / khác): L2A
- Tài khoản GitHub: thanhduc270998
- Repo bài nộp (URL): https://github.com/thanhduc270998/K5-L2A-DAY08-2A202603034-TruongThanhDuc-AIPrelabel.git
- Commit chốt bài (hash): a61587e
- Máy: Windows 10 Home 64-bit, NVIDIA GeForce MX150 2GB + Intel UHD Graphics 620 (chạy model trên CPU qua PyTorch 2.15.0+cpu)
- CVAT (Docker): v2.74.1 (http://localhost:8080)

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 249/326 (76.4%) / 37/326 (11.3%) / 39/326 (12.0%) (mẫu số: 326 box máy); added: 128/415 (30.8%) (mẫu số: 415 đối tượng)
- So với gold: precision / recall của **máy**: 37/43 = 86.0% (KTC 95%: 73–93%) / 37/52 = 71.2% (KTC 95%: 58–82%)
- Bắt bẫy: 11/12 = 91.7% (vòng 1 đạt 8/12 = 66.7%; sau duyệt lại vòng 2 đạt 11/12)
- Thời gian: gán tay 6.25 phút/ảnh (25 phút / 4 ảnh); duyệt có pre-label 2.79 phút/ảnh (67 phút / 24 ảnh)
- Lỗi QC: 100.0% theo ảnh (5/5), 11.4% theo đối tượng (9/79)
- **Quyết định:** Rework — lý do một dòng: Mẫu QC có 5/5 ảnh lỗi (100.0% > 2%, cận trên Wilson 95%: 100%) và lỗi hệ thống `sai_vi_tri_kich_thuoc` (7/9 lỗi) nên cần sửa lại viền/nhãn trên CVAT theo guideline v1.1 trước khi phát hành.
