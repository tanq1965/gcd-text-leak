# N9 — Bốn loại lỗi phân cụm (RQ4)

Theo SimGCD. Chỉ ảnh không nhãn; dùng đúng phép ghép Hungarian của ACC v2 (một ánh xạ cụm → lớp). Tỉ lệ % trên số ảnh Old (hoặc New) không nhãn, nên mỗi nhóm: đúng + nhầm cùng phía + nhầm chéo phía = 100, và cột *đúng* bằng ACC Old/New của N5/N7 (script kiểm khớp từng file). Trung bình ± std qua 5 seed (ddof=1).

## Bảng 1 — Thành phần lỗi

| Tập | Bản | Old: đúng | Old → Old khác | Old → New | New: đúng | New → Old | New → New khác |
|---|---|---:|---:|---:|---:|---:|---:|
| CUB | A | 49.0 ± 1.6 | 11.9 ± 0.7 | 39.1 ± 2.0 | 49.4 ± 1.7 | 21.0 ± 2.5 | 29.6 ± 1.2 |
| CUB | A+B | 57.3 ± 2.1 | 6.1 ± 0.5 | 36.6 ± 2.5 | 69.9 ± 1.4 | 8.7 ± 0.7 | 21.4 ± 1.4 |
| CUB | A+C | 57.1 ± 2.9 | 5.8 ± 0.4 | 37.2 ± 2.7 | 56.8 ± 0.8 | 14.2 ± 1.7 | 29.0 ± 1.4 |
| Stanford Cars | A | 50.6 ± 1.0 | 11.1 ± 0.8 | 38.4 ± 1.3 | 40.2 ± 1.9 | 13.1 ± 0.9 | 46.7 ± 1.4 |
| Stanford Cars | A+B | 71.4 ± 1.9 | 4.2 ± 1.3 | 24.3 ± 2.0 | 73.1 ± 2.4 | 4.1 ± 0.7 | 22.7 ± 2.6 |
| Stanford Cars | A+C | 74.2 ± 0.9 | 3.3 ± 0.2 | 22.4 ± 1.0 | 56.9 ± 2.4 | 7.2 ± 1.1 | 36.0 ± 2.8 |
| FGVC-Aircraft | A | 23.8 ± 2.2 | 27.4 ± 1.9 | 48.8 ± 3.7 | 30.3 ± 1.4 | 24.2 ± 2.2 | 45.5 ± 1.6 |
| FGVC-Aircraft | A+B | 37.2 ± 2.1 | 22.1 ± 0.9 | 40.7 ± 2.4 | 48.3 ± 0.5 | 15.1 ± 0.9 | 36.6 ± 0.9 |
| FGVC-Aircraft | A+C | 35.4 ± 2.8 | 21.2 ± 1.3 | 43.5 ± 3.5 | 40.5 ± 1.2 | 17.6 ± 3.6 | 41.9 ± 2.7 |

## Bảng 2 — Thay đổi so với A (điểm %, theo trung bình)

Dương ở cột lỗi = bản ghép text làm loại lỗi đó tăng so với chỉ ảnh.

| Tập | So sánh | Old → Old khác | Old → New | New → Old | New → New khác |
|---|---|---:|---:|---:|---:|
| CUB | A+B − A | -5.8 | -2.5 | -12.3 | -8.1 |
| CUB | A+C − A | -6.1 | -1.9 | -6.8 | -0.6 |
| Stanford Cars | A+B − A | -6.8 | -14.0 | -9.0 | -23.9 |
| Stanford Cars | A+C − A | -7.7 | -15.9 | -5.9 | -10.7 |
| FGVC-Aircraft | A+B − A | -5.3 | -8.1 | -9.1 | -8.9 |
| FGVC-Aircraft | A+C − A | -6.2 | -5.3 | -6.6 | -3.6 |

## Bảng 3 — Liên hệ N8: lỗi theo tag top-1 của ảnh

Ảnh không nhãn, chia theo tag top-1 của RTG (N8): *đúng tên*, *tên lớp khác trong tập*, *không phải tên lớp*. A và A+B dùng tag bản B, A+C dùng tag bản C; A (chỉ ảnh) là mốc so sánh vì không dùng tag. *Theo tag sai*: trong các ảnh có tag top-1 là tên lớp khác, % bị phân vào đúng lớp mà tag đó gọi tên. Trung bình qua seed.

| Tập | Bản | Tag | Ảnh: đúng tên / tên lớp khác / khác | Lỗi % khi tag đúng tên | Lỗi % khi tag là tên lớp khác | Lỗi % khi tag khác | Theo tag sai % |
|---|---|---|---|---:|---:|---:|---:|
| CUB | A | B | 2546 / 472 / 1478 | 44.7 | 71.1 | 54.5 | 21.1 |
| CUB | A+B | B | 2546 / 472 / 1478 | 20.3 | 78.3 | 44.3 | 55.0 |
| CUB | A+C | C | 821 / 511 / 3164 | 28.9 | 61.8 | 43.7 | 31.4 |
| Stanford Cars | A | B | 3999 / 277 / 1868 | 55.8 | 73.7 | 55.1 | 25.0 |
| Stanford Cars | A+B | B | 3999 / 277 / 1868 | 25.3 | 66.9 | 26.0 | 49.0 |
| Stanford Cars | A+C | C | 1291 / 302 / 4551 | 25.0 | 47.2 | 40.4 | 24.3 |
| FGVC-Aircraft | A | B | 1545 / 2180 / 1276 | 65.2 | 78.6 | 68.4 | 6.4 |
| FGVC-Aircraft | A+B | B | 1545 / 2180 / 1276 | 38.8 | 69.1 | 52.2 | 26.7 |
| FGVC-Aircraft | A+C | C | 606 / 1807 / 2588 | 55.6 | 64.7 | 60.1 | 10.2 |

Hình: `figures/n9_errors.png`.
