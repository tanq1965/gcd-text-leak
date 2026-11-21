# N8 — Truy xuất đúng tên và nhận nhầm tên (RQ1)

> ✅ Mọi hàng dùng RTG khớp hash lexicon hiện tại (kiểm tra tự động khi chạy N8).

Tỉ lệ % trên số ảnh của split (train, cả có nhãn và không nhãn). *Đúng top-3*: tên lớp đúng nằm trong top-3 tag. *Nhầm top-1*: tag đứng đầu là tên lớp khác trong cùng tập (tách theo lớp bị gọi tên là Old/New). *Đúng top-3 (có tên)*: chỉ tính ảnh của lớp có tên trong tag của lexicon B. *Rò tiền tố*: top-3 có tag là tiền tố token ngắn hơn của tên lớp đúng (trực tiếp hoặc sau khi bỏ token đầu, vd `canyon` < `gmc canyon …`), không tính tên chính xác; *(chỉ lớp này)*: tiền tố không khớp lớp nào khác trong tập. Hai cột này dùng bộ tách token riêng, độc lập với bộ lọc N3, nên đo được rò tên cấp thô mà các cột khác không thấy.

| Tập | Bản | Nhóm | Ảnh | Đúng top-3 | Đúng top-3 (có tên) | Đúng top-1 | Nhầm top-1 | ↳ gọi tên Old | ↳ gọi tên New | Top-1 không phải tên lớp | Không tag | Thuộc tính đúng tên | Rò tiền tố top-3 | Rò tiền tố (chỉ lớp này) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUB | B | All | 5994 | 69.5 | 88.6 | 56.0 | 11.2 | 3.6 | 7.6 | 32.8 | 0.0 | 60.7 | 0.6 | 0.0 |
| CUB | B | Old | 2997 | 66.8 | 89.1 | 53.2 | 14.2 | 3.6 | 10.6 | 32.6 | 0.0 | 58.3 | 0.7 | 0.0 |
| CUB | B | New | 2997 | 72.3 | 88.1 | 58.8 | 8.2 | 3.6 | 4.6 | 33.0 | 0.0 | 63.2 | 0.4 | 0.0 |
| CUB | C | All | 5994 | 33.8 | 43.1 | 27.8 | 9.6 | 9.6 | 0.0 | 62.6 | 0.0 | 30.0 | 0.9 | 0.0 |
| CUB | C | Old | 2997 | 67.6 | 90.2 | 55.5 | 4.6 | 4.6 | 0.0 | 39.9 | 0.0 | 59.9 | 1.1 | 0.0 |
| CUB | C | New | 2997 | 0.0 | 0.0 | 0.0 | 14.7 | 14.7 | 0.0 | 85.3 | 0.0 | 0.0 | 0.7 | 0.0 |
| Stanford Cars | B | All | 8144 | 92.4 | 92.4 | 65.4 | 4.0 | 1.1 | 2.9 | 30.7 | 0.0 | 94.4 | 75.7 | 50.4 |
| Stanford Cars | B | Old | 4001 | 91.9 | 91.9 | 65.0 | 2.5 | 1.1 | 1.4 | 32.4 | 0.0 | 95.2 | 72.9 | 49.1 |
| Stanford Cars | B | New | 4143 | 92.9 | 92.9 | 65.7 | 5.4 | 1.1 | 4.2 | 28.9 | 0.0 | 93.6 | 78.4 | 51.6 |
| Stanford Cars | C | All | 8144 | 45.3 | 45.3 | 32.4 | 4.0 | 4.0 | 0.0 | 63.6 | 0.0 | 47.5 | 61.8 | 24.2 |
| Stanford Cars | C | Old | 4001 | 92.2 | 92.2 | 65.9 | 1.1 | 1.1 | 0.0 | 33.0 | 0.0 | 96.6 | 74.8 | 49.3 |
| Stanford Cars | C | New | 4143 | 0.0 | 0.0 | 0.0 | 6.7 | 6.7 | 0.0 | 93.3 | 0.0 | 0.0 | 49.2 | 0.0 |
| FGVC-Aircraft | B | All | 6667 | 57.0 | 57.0 | 30.3 | 43.8 | 20.2 | 23.6 | 25.9 | 0.0 | 47.8 | 0.4 | 0.0 |
| FGVC-Aircraft | B | Old | 3332 | 54.9 | 54.9 | 29.4 | 43.5 | 19.7 | 23.7 | 27.1 | 0.0 | 42.1 | 0.7 | 0.0 |
| FGVC-Aircraft | B | New | 3335 | 59.0 | 59.0 | 31.1 | 44.2 | 20.7 | 23.5 | 24.6 | 0.0 | 53.5 | 0.0 | 0.0 |
| FGVC-Aircraft | C | All | 6667 | 34.0 | 34.0 | 17.8 | 34.9 | 34.9 | 0.0 | 47.2 | 0.0 | 26.5 | 0.7 | 0.0 |
| FGVC-Aircraft | C | Old | 3332 | 68.0 | 68.0 | 35.7 | 30.3 | 30.3 | 0.0 | 34.0 | 0.0 | 53.1 | 0.8 | 0.0 |
| FGVC-Aircraft | C | New | 3335 | 0.0 | 0.0 | 0.0 | 39.5 | 39.5 | 0.0 | 60.5 | 0.0 | 0.0 | 0.7 | 0.0 |
