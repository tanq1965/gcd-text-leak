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

### Đường dẫn Kaggle
- Đã đổi thành `/kaggle/input/datasets/<người_đăng>/<tên_bộ>/` (khác hướng dẫn N0).

## Quyết định kỹ thuật N1

### K-means
- `max_iter=200` (lấy từ `bash_scripts/k_means.sh` của GCD), `n_init=10` (mặc định CLI của GCD).
- Bash script của GCD dùng `n_init=100`, nhưng benchmark trên dữ liệu giả cỡ thật (N=6000, D=1024, K=200) ước lượng khoảng 37 phút/lần chạy trên GPU với `n_init=100`, so với khoảng 3,7 phút với `n_init=10`. Lưới N5/N7 có hơn 100 lần chạy nên phải giảm.
- Sẽ so `n_init` 10 với 100 trên CUB bản A ở pilot 13/11.

### ACC `v2`
- Tính theo **mẫu**, không phải trung bình theo lớp (bản trung bình theo lớp là `split_cluster_acc_v2_balanced`).
- Mục 4.5 của báo cáo ghi đúng như vậy.

### Seed của split
- Cố định là 0, vì `subsample_instances` của SimGCD gọi cứng `np.random.seed(0)`.
- Seed của `eval.py` chỉ ảnh hưởng khởi tạo k-means.

### NaN
- `eval.py` báo lỗi nếu tâm cụm bị NaN (cụm rỗng), và ghi `n_empty_clusters` vào JSON.
