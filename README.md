# GCD: text của TextGCD đến từ đâu?

Đồ án tổng hợp môn Trí tuệ nhân tạo (HCMUT), chủ đề Generalized Category Discovery. Phân tích (không huấn luyện mạng) xem lợi ích của nhánh text trong TextGCD đến từ mô tả thuộc tính hay từ việc lexicon chứa sẵn tên lớp.

## Dữ liệu

Ba bộ dữ liệu lấy từ Kaggle:

- CUB-200-2011: `wenewone/cub2002011`
- FGVC-Aircraft: `seryouxblaster764/fgvc-aircraft`
- Stanford Cars: `rickyyyyyyy/torchvision-stanford-cars`

## Các phần của đồ án

- **N1 — Splits và đánh giá GCD:** xem [docs/data_formats.md](docs/data_formats.md).
- **N3 — Phân tích và lọc TextGCD lexicon:** xem [docs/n3_lexicon.md](docs/n3_lexicon.md).
- **Code tái sử dụng:** xem [third_party/SOURCES.md](third_party/SOURCES.md).

## Chạy lại N3

Chạy các lệnh từ thư mục gốc repository. Cần cài các dependency trong `requirements.txt`.

```powershell
# Tái tạo tên lớp CUB và Aircraft. Stanford Cars cần thêm cars_meta.mat:
python .\scripts\prepare_n3_names.py --cars-meta "D:\path\to\devkit\cars_meta.mat"

# Tạo C/D lexicon và coverage cho cả ba dataset:
python .\scripts\n3_lexicon_alias.py --ds cub --names names\cub.csv --tags third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs third_party\textgcd\Lexicon\Lexicon_attributes.csv --d-strip-all --out .
python .\scripts\n3_lexicon_alias.py --ds aircraft --names names\aircraft.csv --tags third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs third_party\textgcd\Lexicon\Lexicon_attributes.csv --d-strip-all --out .
python .\scripts\n3_lexicon_alias.py --ds scars --names names\scars.csv --tags third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs third_party\textgcd\Lexicon\Lexicon_attributes.csv --d-strip-all --out .

# Rà soát substring candidates của các lớp chưa match:
python .\scripts\review_unmatched.py

# Kiểm thử:
python -m pytest -q

# Kiểm tra trước khi commit/push (read-only):
python .\scripts\prepush_n3_check.py
```

Bản C loại tên lớp New theo exact matching và các alias đã được review. Bản D không dùng tag, cắt mọi tiền tố `X which` bằng `--d-strip-all`, sau đó loại các mô tả trùng chính xác và giữ bản ghi đầu tiên. Quy tắc và số liệu audit chi tiết nằm trong [docs/n3_lexicon.md](docs/n3_lexicon.md).

## File lớn

### Pre-extracted Image Features (N4)

- **File:** `feats_n4.zip` (chứa `{ds}_img.npy` và `{ds}_img_ids.npy` cho CUB, Stanford Cars, FGVC-Aircraft).
- **Google Drive:** [Tải tại đây](https://drive.google.com/drive/folders/1iZhZTkanZov6NvgCODRuZwbILuW1HAdr?usp=sharing)
