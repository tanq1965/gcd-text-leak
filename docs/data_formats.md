# Định dạng dữ liệu và quyết định kỹ thuật

## Quy ước file

### `splits/{ds}.csv`
- Cột: `image_id, path, label, is_labeled, is_old`.
- Thứ tự dòng là **thứ tự chuẩn**: ảnh có nhãn trước, sau đó trong mỗi khối sắp theo `image_id`.
- `image_id` là basename của ảnh, trùng với khoá mà RTG dùng.
- `label` là nhãn đã đánh lại, lớp Old đánh số trước.
- Đọc bằng `dtype={"image_id": str}`.

### `splits/{ds}_classmap.csv` (cho N3)
- Cột: `raw_label, label, is_old, class_name`.
- `raw_label` đánh số từ 0.
  - CUB: `raw_label = k` ứng với dòng `k+1` của `classes.txt`.
  - Cars: phần tử thứ `k` của `class_names` trong `cars_meta.mat`.
  - Aircraft: dùng thẳng cột `class_name`, vì thứ tự lớp ở đây có thể khác `variants.txt`.

### Đặc trưng (N4 tạo; N5, N7 dùng)
- Ảnh: `feats/{ds}_img.npy` (N × D, đã chuẩn hoá L2) kèm `feats/{ds}_img_ids.npy`.
- Text: `feats/{ds}_txt_{ver}.npy` (N × D, đã chuẩn hoá L2) kèm `feats/{ds}_txt_{ver}_ids.npy`, với `{ver}` ∈ {B, C, D}.
- Ý nghĩa `{ver}`: B là lexicon gốc; C bỏ tên lớp New; D bỏ tag và tiền tố tên ở thuộc tính. Định nghĩa đầy đủ ở `docs/n3_lexicon.md`.
- File ids là mảng chuỗi (dtype `<U`, không phải object), mỗi phần tử một `image_id`, đọc được bằng `np.load` mặc định (`allow_pickle=False`).
- Thứ tự dòng không bắt buộc: `eval.py` tự sắp theo `splits` và báo lỗi nếu thiếu, thừa hoặc trùng `image_id`.
- Tương thích cũ: nếu không có `_ids.npy`, `eval.py` đọc `X.ids.txt` (mỗi dòng một `image_id`). Nếu có cả hai thì dùng `_ids.npy`. File mới chỉ ghi `_ids.npy`.

### Câu mô tả (N4 dựng; dùng cho N6, N7 và mục 4.2 báo cáo)
- Khuôn theo TextGCD (`third_party/textgcd/data/cub.py`, `stanford_cars.py`): base text, rồi tag, rồi thuộc tính.
- Base text: CUB `A photo of a bird`, Cars `A photo of a car`, Aircraft `A photo of an aircraft`. Aircraft dùng khuôn CUB (TextGCD không có loader Aircraft).
- Tag: bỏ tiền tố `a photo of a `, đi kèm lần lượt `most likely`, `probably`, `perhaps`. Cars thêm chữ `a` trước tag (`most likely a <tag>`).
- Thuộc tính: đi kèm lần lượt `Likely`, `Perhaps`, `Could be`.
- Ví dụ CUB: `A photo of a bird, most likely Bobolink, probably Indigo Bunting, perhaps Painted Bunting. Likely bobolink which has small songbird. Perhaps bobolink which has black and white markings.`
- Ảnh không có tag hoặc thuộc tính nào thì bỏ hẳn phần đó (câu gọn), không giữ dấu câu thừa như code TextGCD. Ví dụ không có cả hai: `A photo of a bird.`. Áp dụng cả cho bản D (không tag).
- `encode_text.py` báo lỗi nếu `image_id` của splits thiếu hẳn trong file RTG, và in số ảnh có `image_id` nhưng không có tag hay thuộc tính nào.

### Kết quả thí nghiệm `results/{ds}_{variant}_s{seed}.json` (N5, N7 tạo)
- `variant` cũng nằm trong `config` của file JSON. Các biến thể:
  - `A` (chỉ ảnh), `B`, `C`, `D` (chỉ text bản tương ứng).
  - `A+B`, `A+C`, `A+D`: ghép với α = 0,5.
  - `A+B_a0.25`, `A+B_a0.75` (tương tự cho C, D): độ nhạy α.
- α ghi trong `config.alpha`; `None` nếu không ghép.
- `pred` có độ dài bằng số hàng của `splits/{ds}.csv`.
- Kiểm bằng `python -m scripts.verify_results` (chạy từ gốc repo).

### Đường dẫn Kaggle
- Đã đổi thành `/kaggle/input/datasets/<người_đăng>/<tên_bộ>/` (khác hướng dẫn N0).

## Quyết định kỹ thuật

### K-means
- `max_iter=200` (lấy từ `bash_scripts/k_means.sh` của GCD), `n_init=10` (mặc định CLI của GCD).
- Bash script của GCD dùng `n_init=100`; với lưới hơn 100 lần chạy thì quá tốn thời gian (ước lượng cũ khoảng 37 phút mỗi lần ở N=6000, D=1024, K=200). Với `n_init=10`, đo thật trên bản A: CUB 712 s, Cars 213 s, Aircraft 269 s mỗi lần chạy.
- So sánh `n_init` 10 với 100 trên CUB bản A: chưa ghi kết quả; hiện giữ 10.

### ACC `v2`
- Tính theo **mẫu**, không phải trung bình theo lớp (bản trung bình theo lớp là `split_cluster_acc_v2_balanced`).
- Phạm vi: K-means chạy trên mọi hàng của split (có nhãn và không nhãn; K = số lớp Old + New). ACC chỉ tính trên các hàng `is_labeled=0`.
- Một phép ghép Hungarian duy nhất trên các hàng `is_labeled=0`; ma trận vuông `D×D`, `D = max(nhãn thật, nhãn cụm) + 1`.
- All, Old, New dùng chung một ánh xạ cụm → lớp. Mẫu số: số hàng không nhãn (All), số hàng `is_old=1` (Old), số hàng `is_old=0` (New). Hàng có nhãn không tham gia ACC.
- Tổng hợp theo seed: trung bình ×100, độ lệch chuẩn mẫu (`ddof=1`); chỉ 1 seed thì std hiển thị 0,00.
- Mục 4.5 của báo cáo ghi đúng như vậy. Định nghĩa được kiểm độc lập bằng `scripts/verify_results.py`.

### Seed của split
- Cố định là 0, vì `subsample_instances` của SimGCD gọi cứng `np.random.seed(0)`.
- Seed của `eval.py` chỉ ảnh hưởng khởi tạo k-means.

### NaN
- `eval.py` báo lỗi nếu tâm cụm bị NaN (cụm rỗng), và ghi `n_empty_clusters` vào JSON.
