# N3 — Phân tích và lọc lexicon TextGCD

## 1. Mục tiêu và phạm vi

N3 kiểm tra xem lexicon của TextGCD đã chứa tên lớp của CUB, FGVC-Aircraft và Stanford Cars hay chưa, sau đó tạo hai điều kiện lexicon để tách ảnh hưởng của tên lớp khỏi mô tả thuộc tính.

N3 là bước phân tích dữ liệu, không huấn luyện mô hình.

## 2. Đầu vào và quy ước

### Danh sách lớp

`names/{ds}.csv` có đúng bốn cột:

```text
label,class_name,name_norm,is_old
```

- `label` là nhãn đã ánh xạ theo `splits/{ds}.csv`, bắt đầu từ 0 và liên tục, theo relabeling SimGCD/SSB (seed 0).
- Lớp Old đứng trước lớp New, theo mapping Old/New được nhóm xác nhận.
- `name_norm` là tên chuẩn hoá dùng để so khớp.
- `is_old` nhận `1` hoặc `0`.

Số lớp phải đúng:

| Dataset | Tổng lớp | Old | New |
|---|---:|---:|---:|
| CUB-200-2011 | 200 | 100 | 100 |
| FGVC-Aircraft | 100 | 50 | 50 |
| Stanford Cars | 196 | 98 | 98 |

Nguồn tên lớp:
- CUB: tên thư mục lớp trong `splits/cub.csv`, đối chiếu `splits/cub_classmap.csv`.
- Aircraft: `class_name` trong `splits/aircraft_classmap.csv`, không lấy thứ tự từ `variants.txt`.
- Cars: `class_names` trong `cars_meta.mat`, ánh xạ qua `splits/scars_classmap.csv`.

### Lexicon gốc

Nguồn là hai file trong `third_party/textgcd/Lexicon/`:

- `Lexicon_tags.csv`
- `Lexicon_attributes.csv`

Đây là CSV **một cột, không có header**. Code RTG gốc đọc mỗi dòng bằng `csv.reader` và lấy `row[0]`; vì vậy không được thêm header vào lexicon đầu ra.

Các file audit (`removed.csv`, `modified.csv`) và coverage là CSV có header.

## 3. Định nghĩa các điều kiện

### B — Baseline

Dùng nguyên hai file lexicon gốc của TextGCD.

### C — Loại tên lớp New

- Xoá tag nếu phần tên lớp sau tiền tố `a photo of a/an` khớp chính xác với tên lớp New hoặc alias đã duyệt.
- Với thuộc tính dạng `X which Y`, nếu `X` là tên lớp New hoặc alias đã duyệt thì chỉ giữ phần mô tả `Y`.
- Giữ tag và prefix thuộc tính của lớp Old.
- Nếu tên lớp New còn trong mô tả ở dòng đã review, xoá toàn bộ thuộc tính và ghi `rule=residual_name` vào `C/removed.csv`.
- Không dùng substring matching để tự động quyết định alias.

### D — Không dùng tag, loại prefix tên lớp của dataset

- `tags.csv` là file **0 byte** (không có dòng, không có header).
- Với thuộc tính dạng `X which Y`, bỏ `X` nếu `X` khớp tên lớp hoặc alias của bất kỳ lớp nào thuộc dataset đang xét.
- Prefix thuộc các lớp không thuộc dataset được giữ nguyên. Không bật `--d-strip-all` trong kết quả hiện tại.
- Các dòng thuộc tính còn tên lớp trong mô tả đã được review sẽ bị xoá toàn bộ và ghi `rule=residual_name` vào `D/removed.csv`.
- `D/removed.csv` chỉ liệt kê thuộc tính bị xoá; tổng số 22.131 tag bị xoá được ghi trong audit JSON.

Các dòng thuộc tính đã được xác nhận cần xoá (số dòng nguồn trong `Lexicon_attributes.csv`):

| Phiên bản | Dòng nguồn | Lớp |
|---|---:|---|
| Aircraft C | 17747 | ERJ 135 (New) |
| Aircraft D | 17747 | ERJ 135 (New) |
| Aircraft D | 17585 | Cessna 172 (Old) |

**Lưu ý tích hợp N2/N6:** RTG gốc chưa hỗ trợ danh sách tag rỗng. Với D, N2/N6 cần bỏ qua bước tag embedding, xử lý top-k retrieval và `outputs['tags']` tương ứng, nhưng vẫn xử lý attributes. Không thêm dòng giả hoặc header vào `D/tags.csv` để né lỗi.

## 4. Mapping alias đã được xác nhận

Các alias là quy tắc so khớp **chính xác**, không phải luật chứa chuỗi con.

### CUB

| Lớp CUB | Alias | Lý do trong audit |
|---|---|---|
| `105.Whip_poor_Will` | Eastern Whip-poor-will | `AOU 2010 split` |
| `105.Whip_poor_Will` | Mexican Whip-poor-will | `AOU 2010 split` |
| `091.Mockingbird` | Northern Mockingbird | `only common North American mockingbird` |
| `101.White_Pelican` | American White Pelican | `CUB White Pelican species name` |
| `017.Cardinal` | Northern Cardinal | `CUB Cardinal species name` |

- Trong lexicon hiện tại có tag `Eastern Whip poor will`; chưa tìm thấy dòng `Mexican Whip-poor-will`. Alias Mexican vẫn được khai báo để quy tắc mapping đầy đủ.
- `022.Chuck_will_Widow` là lớp riêng, không phải alias của `Whip_poor_Will`.
- Các loài mockingbird khác (ví dụ Tropical, Bahama) không được gộp vào `Mockingbird`.
- Giữ `cardinal directions` (attribute) và `Blue cardinal flower` (tag) vì khác nghĩa.

### Aircraft

| Lớp Aircraft | Alias |
|---|---|
| `A320` | Airbus A320 family |
| `DC-8` | Douglas DC 8 |
| `ERJ 145` | Embraer ERJ 145 family |
| `MD-80` | McDonnell Douglas MD 80 |
| `747-400` | Boeing 747 400 |
| `C-130` | Lockheed C 130 Hercules |
| `DC-6` | Douglas DC 6 |
| `Challenger 600` | Bombardier Challenger 600 |
| `Metroliner` | Fairchild C 26 Metroliner |

Lý do từng alias (tên nhà sản xuất, tên family hoặc biến thể quân sự) được ghi trong `ALIAS_REASONS` và audit JSON.

`North American B-45 Tornado` là máy bay khác với `Tornado`, nên được giữ lại.

Trong `removed.csv`/`modified.csv`, dòng alias có `rule=alias` và `reason` tương ứng. Dòng thuộc tính bị xoá vì còn tên lớp trong mô tả sử dụng `rule=residual_name`.

## 5. Kết quả đã tái tạo

Coverage bên dưới được tính trên **lexicon gốc B**: một lớp được tính là có mặt nếu tên chuẩn hoặc alias chính xác xuất hiện trong tag, hoặc là prefix `X` của thuộc tính `X which Y`. Đây không phải tỷ lệ coverage sau lọc C/D.

| Dataset | Nhóm | Tag | Attribute | Either |
|---|---|---:|---:|---:|
| CUB | All (200) | 71.5% | 70.5% | 71.5% |
| CUB | Old (100) | 70.0% | 69.0% | 70.0% |
| CUB | New (100) | 73.0% | 72.0% | 73.0% |
| Aircraft | All (100) | 100.0% | 100.0% | 100.0% |
| Stanford Cars | All (196) | 100.0% | 100.0% | 100.0% |

Audit counts sau review:

| Dataset | C removed tags | C removed attrs | C modified attrs | D removed tags | D removed attrs | D modified attrs |
|---|---:|---:|---:|---:|---:|---:|
| CUB | 75 | 0 | 548 | 22,131 | 0 | 1,082 |
| Aircraft | 62 | 1 | 401 | 22,131 | 2 | 805 |
| Stanford Cars | 98 | 0 | 845 | 22,131 | 0 | 1,712 |

Sau alias mapping, CUB còn 57 lớp không match theo tên chính xác/alias đã duyệt. `coverage/cub_unmatched_candidates.csv` tìm thấy 3 nhóm có substring hit:
- `Tree_Sparrow` ↔ `American Tree Sparrow`
- `Frigatebird` ↔ `Magnificent Frigatebird`
- `Nighthawk` ↔ `Common Nighthawk`

Đây chỉ là ứng viên để người làm đồ án xem xét; không tự động gộp vì tên gần nhau không chứng minh cùng loài. Báo cáo đầy đủ nằm ở `coverage/cub_unmatched_review.txt`. Aircraft và Stanford Cars không còn lớp unmatched theo phép so khớp hiện tại.

## 6. Đầu ra

Với mỗi dataset:

```text
coverage/{ds}.csv
coverage/{ds}_summary.csv
coverage/{ds}_audit.json
coverage/{ds}_unmatched_candidates.csv

lexicon/{ds}/C/
  tags.csv
  attributes.csv
  removed.csv
  modified.csv

lexicon/{ds}/D/
  tags.csv
  attributes.csv
  removed.csv
  modified.csv
```

- `coverage/{ds}.csv`: kết quả match từng lớp trên lexicon gốc.
- `*_summary.csv`: tổng hợp All/Old/New.
- `*_audit.json`: số dòng đầu vào/đầu ra, mapping alias, lớp unmatched, số dòng xoá/sửa và phạm vi D.
- `removed.csv`: các dòng bị xoá, gồm file nguồn, số dòng gốc, nội dung, lớp match và rule.
- `modified.csv`: các dòng thuộc tính được đổi, gồm nội dung cũ/mới, lớp match và rule.
- Riêng `D/removed.csv` chỉ giữ các dòng thuộc tính bị xoá (`residual_name`); số tag bị xoá được lưu trong `D_removed_tags` của audit JSON.

Audit JSON sử dụng newline `\n` để kết quả tái lập nhất quán trên Windows/Linux.

## 7. Tái tạo

Chạy từ thư mục gốc repository. Cần các package trong `requirements.txt`.

```powershell
# Chỉ cần chạy nếu muốn tạo lại names/.
# Stanford Cars cần cars_meta.mat từ devkit.
python .\scripts\prepare_n3_names.py --cars-meta "D:\path\to\devkit\cars_meta.mat"

# Chạy từng dataset; --out . ghi kết quả vào coverage/ và lexicon/.
python .\scripts\n3_lexicon_alias.py --ds cub --names .\names\cub.csv --tags .\third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs .\third_party\textgcd\Lexicon\Lexicon_attributes.csv --out .
python .\scripts\n3_lexicon_alias.py --ds aircraft --names .\names\aircraft.csv --tags .\third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs .\third_party\textgcd\Lexicon\Lexicon_attributes.csv --out .
python .\scripts\n3_lexicon_alias.py --ds scars --names .\names\scars.csv --tags .\third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs .\third_party\textgcd\Lexicon\Lexicon_attributes.csv --out .

# Sinh báo cáo substring candidates để review thủ công.
python .\scripts\review_unmatched.py

# Kiểm thử toàn repository.
python -m pytest -q

# Pre-push checks (read-only; không tự add/commit/push).
python .\scripts\prepush_n3_check.py
```

Không cần `cars_meta.mat` để dùng các file `names/scars.csv` đã commit; file này chỉ cần nếu muốn tái tạo danh sách tên lớp Cars từ devkit gốc.

Không sửa tay CSV đầu ra; thay đổi alias hoặc quy tắc lọc phải thực hiện trong script rồi tái tạo kết quả.

## 8. Giới hạn cần ghi trong báo cáo

- Quy tắc tự động chỉ nhận diện tên tag chính xác và prefix thuộc tính `X which Y`; nó không chứng minh mọi cách nhắc tên lớp trong phần mô tả tự do đã được loại bỏ.
- Chỉ xử lý `residual_name` tại các dòng đã được review; không tự động xoá mọi dòng chứa tên lớp.
- Các substring candidates không được tự động coi là alias.
- Các tên dài hơn như `American Tree Sparrow`, `Common/Lesser Nighthawk`, `Magnificent Frigatebird` và các mẫu xe Old khác năm trong Cars giữ nguyên theo phạm vi đã thống nhất.
- `North American B-45 Tornado` khác với lớp `Tornado`, không được coi là alias.
- Cần N2/N6 xử lý riêng danh sách tag rỗng của D như lưu ý ở mục 3.
- N8 có thể dùng chung hàm `aliases()` của N3 để đánh giá truy xuất đúng tên lớp.
- TextGCD upstream không có file LICENSE trong snapshot được sử dụng; provenance và ghi chú giấy phép được ghi tại `third_party/SOURCES.md`.