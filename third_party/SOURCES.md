# Nguồn code tái sử dụng

Mọi file `.py` copy từ repo gốc có dòng đầu (hoặc dòng 2 nếu file có shebang/khai báo coding):
`# Source: <repo>@<commit-ngắn>, <đường dẫn gốc>`. Logic không bị sửa. Các dòng import đã đổi để chạy được với thư viện mới được liệt kê ở mục "Thay đổi so với bản gốc".

| Repo | URL | Commit đầy đủ | License | Ngày clone |
|---|---|---|---|---|
| simgcd | https://github.com/CVMI-Lab/SimGCD | f843f27195701b7c8c5713cfd17499ad67c86247 | MIT (third_party/simgcd/LICENSE) | 8/11 |
| gcd | https://github.com/sgvaze/generalized-category-discovery | 831a645c3d09a68ec4633a45741025765bacf7e0 | MIT (third_party/gcd/LICENSE) | 8/11 |
| textgcd | https://github.com/HaiyangZheng/TextGCD | 1175472e150dea8f68c9a02f426aa0ba4ea52819 | Không có file LICENSE | 8/11 |

## Ghi chú

- TextGCD không có file LICENSE trong repo gốc. Code được dùng cho mục đích học tập, có ghi nguồn đầy đủ.
- `ssb_splits/*.pkl` của SimGCD và TextGCD trùng md5 (kiểm tra 8/11):
  - `aircraft_osr_splits.pkl`: `e8f9c62211711d9c5e84ead6b205e4c4`
  - `cub_osr_splits.pkl`: `a12ff520ed8777b156d8202e30f6a0c2`
  - `herbarium_19_class_splits.pkl`: `2503f428cb7693d84fba417a4d719ebb`
  - `scars_osr_splits.pkl`: `e264b7c905095ce133b3f638b2151718`
- Không đưa `third_party/simgcd` và `third_party/textgcd` vào `sys.path` cùng lúc (trùng module `data`, `config`).
- File trong repo lưu xuống dòng LF (theo .gitattributes); bản checkout trên Windows của repo gốc là CRLF.
- Phụ thuộc: gcd `project_utils/cluster_utils.py` import matplotlib; textgcd `utils/__init__.py` import tensorboardX (qua general_utils).
- `__init__.py` rỗng do nhóm thêm (không có trong repo gốc, không có dòng Source): `simgcd/data/`, `simgcd/util/`, `gcd/methods/`, `gcd/methods/clustering/`, `textgcd/data/`.

## Danh sách file đã copy

**simgcd** (`third_party/simgcd/`)
- `data/` (cả thư mục, gồm `ssb_splits/`)
- `util/cluster_and_log_utils.py`
- `config.py`
- `LICENSE`

**gcd** (`third_party/gcd/`)
- `methods/clustering/faster_mix_k_means_pytorch.py`
- `project_utils/__init__.py`
- `project_utils/cluster_utils.py`
- `LICENSE`

**textgcd** (`third_party/textgcd/`)
- `retrieval_based_text_generation.py`
- `config.py`
- `data/`
- `clip/`
- `utils/`
- `Lexicon/`

## Thay đổi so với bản gốc

- gcd `methods/clustering/faster_mix_k_means_pytorch.py` dòng import: `sklearn.utils._joblib` → `joblib` (module không còn trong sklearn bản mới; đã kiểm tra trên sklearn 1.9.1). Không đổi logic.

## Tham khảo, không copy

Các file này import module của repo GCD mà nhóm không copy (data/, config, project_utils.general_utils, ...), nên không đưa vào third_party. Chỉ đọc để tham khảo:
- GCD `methods/clustering/k_means.py`: cách gọi `K_Means.fit_mix` và tách dự đoán phần không nhãn (N1). https://github.com/sgvaze/generalized-category-discovery/blob/831a645c3d09a68ec4633a45741025765bacf7e0/methods/clustering/k_means.py
- GCD `methods/clustering/extract_features.py`: mẫu vòng lặp trích đặc trưng (N4). https://github.com/sgvaze/generalized-category-discovery/blob/831a645c3d09a68ec4633a45741025765bacf7e0/methods/clustering/extract_features.py
