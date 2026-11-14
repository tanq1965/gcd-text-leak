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

### D — Không dùng tag, loại mọi prefix và bỏ thuộc tính trùng

- `tags.csv` là file **0 byte**, không có dòng hoặc header.
- Bật `--d-strip-all`: với mọi thuộc tính dạng `X which Y`, bỏ tiền tố `X which` và giữ mô tả `Y`, kể cả khi `X` không thuộc dataset đang xét.
- Sau khi cắt tiền tố, loại các mô tả trùng chính xác; giữ lần xuất hiện đầu tiên theo thứ tự nguồn.
- Các dòng bị loại vì trùng được ghi vào `D/removed.csv` với `rule=D_deduplicate_after_strip`, số dòng nguồn và lý do.
- Các dòng còn chứa tên lớp trong mô tả đã được review được xóa toàn bộ với `rule=residual_name`.
- `D/modified.csv` chỉ ghi những thuộc tính được sửa và thực sự giữ lại sau bước loại trùng.
- Audit JSON ghi `D_removed_attributes`, `D_deduplicated_attributes`, `D_removed_residual_attributes` và `D_modified_attributes`.
- `D_scope` là `all_attribute_prefixes`; số lượng 22.131 tag bị loại được ghi ở `D_removed_tags`, không liệt kê lại trong `D/removed.csv`.
- Quy tắc này loại bỏ các prefix dạng `X which`, nhưng không chứng minh mọi tên lớp nằm sâu trong văn bản tự do đều đã biến mất.

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
| `067.Anna_Hummingbird` | Annas Hummingbird | `possessive/typo spelling` |
| `193.Bewick_Wren` | Bewicks Wren | `possessive/typo spelling` |
| `193.Bewick_Wren` | Bewick s Wren | `possessive/typo spelling` |
| `023.Brandt_Cormorant` | Brandts Cormorant | `possessive/typo spelling` |
| `009.Brewer_Blackbird` | Brewers Blackbird | `possessive/typo spelling` |
| `009.Brewer_Blackbird` | Brewer's Blackbird | `possessive/typo spelling` |
| `115.Brewer_Sparrow` | Brewers Sparrow | `possessive/typo spelling` |
| `093.Clark_Nutcracker` | Clarks Nutcracker | `possessive/typo spelling` |
| `061.Heermann_Gull` | Heermanns Gull | `possessive/typo spelling` |
| `123.Henslow_Sparrow` | Henslows Sparrow | `possessive/typo spelling` |
| `146.Forsters_Tern` | Forster s Tern | `possessive/typo spelling` |
| `141.Artic_Tern` | Arctic Tern | `possessive/typo spelling` |
| `124.Le_Conte_Sparrow` | Le Contes Sparrow | `possessive/typo spelling` |
| `125.Lincoln_Sparrow` | Lincolns Sparrow | `possessive/typo spelling` |
| `098.Scott_Oriole` | Scotts Oriole | `possessive/typo spelling` |
| `180.Wilson_Warbler` | Wilsons Warbler | `possessive/typo spelling` |
| `122.Harris_Sparrow` | Harriss Sparrow | `possessive/typo spelling` |

- Trong lexicon hiện tại có tag `Eastern Whip poor will`; chưa tìm thấy dòng `Mexican Whip-poor-will`. Alias Mexican vẫn được khai báo để quy tắc mapping đầy đủ.
- `022.Chuck_will_Widow` là lớp riêng, không phải alias của `Whip_poor_Will`.
- Các loài mockingbird khác (ví dụ Tropical, Bahama) không được gộp vào `Mockingbird`.
- Giữ `cardinal directions` (attribute) và `Blue cardinal flower` (tag) vì khác nghĩa.

Các alias bổ sung gồm 10 lớp New và 5 lớp Old (17 cách viết, thuộc 15 lớp). Các alias được duyệt theo đúng tên đầy đủ; không dùng substring matching hoặc tự động xóa dấu câu để quyết định alias.

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
| CUB | All (200) | 78.5% | 77.5% | 78.5% |
| CUB | Old (100) | 75.0% | 74.0% | 75.0% |
| CUB | New (100) | 82.0% | 81.0% | 82.0% |
| Aircraft | All (100) | 100.0% | 100.0% | 100.0% |
| Stanford Cars | All (196) | 100.0% | 100.0% | 100.0% |

Audit counts sau review:

| Dataset | C removed tags | C removed attrs | C modified attrs | D removed tags | D removed attrs | D deduplicated attrs | D modified attrs |
|---|---:|---:|---:|---:|---:|---:|---:|
| CUB | 87 | 0 | 617 | 22,131 | 6,739 | 6,739 | 27,994 |
| Aircraft | 62 | 1 | 401 | 22,131 | 6,741 | 6,739 | 27,992 |
| Stanford Cars | 98 | 0 | 845 | 22,131 | 6,739 | 6,739 | 27,994 |

Sau alias mapping, CUB còn 43 lớp không match theo tên chính xác/alias đã duyệt. `coverage/cub_unmatched_candidates.csv` tìm thấy 3 nhóm có substring hit:
- `Tree_Sparrow` ↔ `American Tree Sparrow`
- `Frigatebird` ↔ `Magnificent Frigatebird`
- `Nighthawk` ↔ `Common Nighthawk`

Đây chỉ là ứng viên để người làm đồ án xem xét; không tự động gộp vì tên gần nhau không chứng minh cùng loài. Danh sách ứng viên cập nhật nằm ở `coverage/cub_unmatched_candidates.csv`; `coverage/cub_unmatched_review.txt` là ghi chú review thủ công. Aircraft và Stanford Cars không còn lớp unmatched theo phép so khớp hiện tại.

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
- Riêng `D/removed.csv` ghi các thuộc tính bị xóa do `residual_name` hoặc `D_deduplicate_after_strip`; số tag bị loại được lưu trong `D_removed_tags` của audit JSON.

Audit JSON sử dụng newline `\n` để kết quả tái lập nhất quán trên Windows/Linux.

Audit JSON không có danh sách đầy đủ các lớp bị ảnh hưởng hoặc chỉ báo `">1/2 thuộc tính"`; xem cột `matched_class` trong `removed.csv` và `modified.csv` để truy vết các bản ghi có ánh xạ lớp.

## 7. Tái tạo

Chạy từ thư mục gốc repository. Cần các package trong `requirements.txt`.

```powershell
# Chỉ cần chạy nếu muốn tạo lại names/.
# Stanford Cars cần cars_meta.mat từ devkit.
python .\scripts\prepare_n3_names.py --cars-meta "D:\path\to\devkit\cars_meta.mat"

# Chạy từng dataset; --out . ghi kết quả vào coverage/ và lexicon/.
python .\scripts\n3_lexicon_alias.py --ds cub --names .\names\cub.csv --tags .\third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs .\third_party\textgcd\Lexicon\Lexicon_attributes.csv --d-strip-all --out .
python .\scripts\n3_lexicon_alias.py --ds aircraft --names .\names\aircraft.csv --tags .\third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs .\third_party\textgcd\Lexicon\Lexicon_attributes.csv --d-strip-all --out .
python .\scripts\n3_lexicon_alias.py --ds scars --names .\names\scars.csv --tags .\third_party\textgcd\Lexicon\Lexicon_tags.csv --attrs .\third_party\textgcd\Lexicon\Lexicon_attributes.csv --d-strip-all --out .

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
- Loại trùng ở D chỉ dựa trên văn bản giống hệt sau khi cắt tiền tố; không gộp những câu chỉ tương đương về ngữ nghĩa.
- Cần N2/N6 xử lý riêng danh sách tag rỗng của D như lưu ý ở mục 3.
- N8 có thể dùng chung hàm `aliases()` của N3 để đánh giá truy xuất đúng tên lớp.
- TextGCD upstream không có file LICENSE trong snapshot được sử dụng; provenance và ghi chú giấy phép được ghi tại `third_party/SOURCES.md`.
