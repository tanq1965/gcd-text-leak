# N8 — Truy xuất đúng tên và nhận nhầm tên (RQ1)

> ✅ Mọi hàng dùng RTG khớp hash lexicon hiện tại (kiểm tra tự động khi chạy N8).

## Bảng 1 — Tên chính xác / alias

Tỉ lệ % trên số ảnh của split (train, cả có nhãn và không nhãn). *Đúng top-3*: tên lớp đúng nằm trong top-3 tag. *Nhầm top-1*: tag đứng đầu là tên lớp khác trong cùng tập (tách theo lớp bị gọi tên là Old/New). *Đúng top-3 (có tên)*: chỉ tính ảnh của lớp có tên trong tag của lexicon B.

| Tập | Bản | Nhóm | Ảnh | Đúng top-3 | Đúng top-3 (có tên) | Đúng top-1 | Nhầm top-1 | ↳ gọi tên Old | ↳ gọi tên New | Top-1 không phải tên lớp | Không tag | Thuộc tính đúng tên |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUB | B | All | 5994 | 69.5 | 88.6 | 56.0 | 11.2 | 3.6 | 7.6 | 32.8 | 0.0 | 60.7 |
| CUB | B | Old | 2997 | 66.8 | 89.1 | 53.2 | 14.2 | 3.6 | 10.6 | 32.6 | 0.0 | 58.3 |
| CUB | B | New | 2997 | 72.3 | 88.1 | 58.8 | 8.2 | 3.6 | 4.6 | 33.0 | 0.0 | 63.2 |
| CUB | C | All | 5994 | 33.8 | 43.1 | 27.8 | 9.6 | 9.6 | 0.0 | 62.6 | 0.0 | 30.0 |
| CUB | C | Old | 2997 | 67.6 | 90.2 | 55.5 | 4.6 | 4.6 | 0.0 | 39.9 | 0.0 | 59.9 |
| CUB | C | New | 2997 | 0.0 | 0.0 | 0.0 | 14.7 | 14.7 | 0.0 | 85.3 | 0.0 | 0.0 |
| Stanford Cars | B | All | 8144 | 92.4 | 92.4 | 65.4 | 4.0 | 1.1 | 2.9 | 30.7 | 0.0 | 94.4 |
| Stanford Cars | B | Old | 4001 | 91.9 | 91.9 | 65.0 | 2.5 | 1.1 | 1.4 | 32.4 | 0.0 | 95.2 |
| Stanford Cars | B | New | 4143 | 92.9 | 92.9 | 65.7 | 5.4 | 1.1 | 4.2 | 28.9 | 0.0 | 93.6 |
| Stanford Cars | C | All | 8144 | 45.3 | 45.3 | 32.4 | 4.0 | 4.0 | 0.0 | 63.6 | 0.0 | 47.5 |
| Stanford Cars | C | Old | 4001 | 92.2 | 92.2 | 65.9 | 1.1 | 1.1 | 0.0 | 33.0 | 0.0 | 96.6 |
| Stanford Cars | C | New | 4143 | 0.0 | 0.0 | 0.0 | 6.7 | 6.7 | 0.0 | 93.3 | 0.0 | 0.0 |
| FGVC-Aircraft | B | All | 6667 | 57.0 | 57.0 | 30.3 | 43.8 | 20.2 | 23.6 | 25.9 | 0.0 | 47.8 |
| FGVC-Aircraft | B | Old | 3332 | 54.9 | 54.9 | 29.4 | 43.5 | 19.7 | 23.7 | 27.1 | 0.0 | 42.1 |
| FGVC-Aircraft | B | New | 3335 | 59.0 | 59.0 | 31.1 | 44.2 | 20.7 | 23.5 | 24.6 | 0.0 | 53.5 |
| FGVC-Aircraft | C | All | 6667 | 34.0 | 34.0 | 17.8 | 34.9 | 34.9 | 0.0 | 47.2 | 0.0 | 26.5 |
| FGVC-Aircraft | C | Old | 3332 | 68.0 | 68.0 | 35.7 | 30.3 | 30.3 | 0.0 | 34.0 | 0.0 | 53.1 |
| FGVC-Aircraft | C | New | 3335 | 0.0 | 0.0 | 0.0 | 39.5 | 39.5 | 0.0 | 60.5 | 0.0 | 0.0 |

## Bảng 2 — Rò tên cấp thô (tiền tố token)

Đo độc lập với bộ lọc N3 (bộ tách token riêng). Một tag (hoặc phần `X` của `X which …`) *rò tiền tố* nếu là tiền tố token ngắn hơn của tên/alias lớp đúng; chỉ bỏ **tên hãng** ở đầu (Cars: hãng xe; Aircraft: hãng trong alias đã duyệt; CUB: không bỏ), nên `canyon` < `gmc canyon …`, `boeing 747` khớp cả 747-100/200 (Old) lẫn 747-400 (New), còn `Manic GT` không khớp `Ford GT`. Không tính tên chính xác và dạng `từ thêm + tên chính xác` (`lesser nighthawk`). *≥2 token* / *1 token* tách theo số token của tag (nhóm 1 token gồm cả từ thông dụng như `house`, cần kiểm tay: `n8_prefix_hits.csv`). *Chỉ-New*: mọi lớp khớp đều New (luật N3 của Tân; ảnh Old luôn 0). *Chỉ lớp này*: chỉ khớp đúng lớp của ảnh (chặt hơn).

| Tập | Bản | Nhóm | Tag ≥2 token | Tag 1 token (kiểm tay) | Tag chỉ-New | Tag chỉ lớp này | Thuộc tính | Thuộc tính chỉ-New | Thuộc tính chỉ lớp này |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CUB | B | All | 0.0 | 0.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CUB | B | Old | 0.0 | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CUB | B | New | 0.0 | 0.4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CUB | C | All | 0.0 | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CUB | C | Old | 0.0 | 1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| CUB | C | New | 0.0 | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Stanford Cars | B | All | 74.2 | 13.1 | 28.7 | 50.6 | 0.1 | 0.0 | 0.0 |
| Stanford Cars | B | Old | 70.9 | 14.4 | 0.0 | 49.9 | 0.1 | 0.0 | 0.0 |
| Stanford Cars | B | New | 77.4 | 11.8 | 56.5 | 51.4 | 0.0 | 0.0 | 0.0 |
| Stanford Cars | C | All | 47.5 | 24.9 | 0.5 | 25.1 | 0.5 | 0.4 | 0.4 |
| Stanford Cars | C | Old | 72.2 | 16.2 | 0.0 | 50.0 | 0.1 | 0.0 | 0.0 |
| Stanford Cars | C | New | 23.6 | 33.3 | 1.0 | 1.0 | 1.0 | 0.8 | 0.8 |
| FGVC-Aircraft | B | All | 5.0 | 0.4 | 0.0 | 1.5 | 0.0 | 0.0 | 0.0 |
| FGVC-Aircraft | B | Old | 6.9 | 0.7 | 0.0 | 2.9 | 0.0 | 0.0 | 0.0 |
| FGVC-Aircraft | B | New | 3.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| FGVC-Aircraft | C | All | 6.5 | 0.6 | 0.0 | 1.9 | 0.0 | 0.0 | 0.0 |
| FGVC-Aircraft | C | Old | 8.9 | 0.8 | 0.0 | 3.8 | 0.0 | 0.0 | 0.0 |
| FGVC-Aircraft | C | New | 4.0 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
