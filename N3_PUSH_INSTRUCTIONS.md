# N3 – Lưu ý khi chạy

Một số lưu ý khi chạy lại N3:

1. **Thư mục chạy:** Chạy các script từ thư mục gốc của repository `gcd-text-leak`. Đảm bảo đã kích hoạt virtual environment và cài đủ dependencies như hướng dẫn trong README.

2. **Dữ liệu đầu vào:** Kiểm tra hai file `Lexicon_tags.csv` và `Lexicon_attributes.csv` trong `third_party/textgcd/Lexicon/`. Các file `names/*.csv` cần có đủ bốn cột `label`, `class_name`, `name_norm`, `is_old`.

3. **Mapping Old/New:** `label` phải theo thứ tự relabeling của SimGCD (SSB, seed 0), còn `is_old` phải theo mapping Old/New do Tân cung cấp. Không suy luận Old/New từ thứ tự tên lớp trong dataset.

4. **Tên lớp và alias:** CUB cần bỏ tiền tố số thứ tự khỏi tên lớp khi đối chiếu. Aircraft có thể cần đối chiếu thêm hậu tố `aircraft`. Với CUB, hai trường hợp Whip-poor-will và Mockingbird đã được thống nhất xử lý dưới dạng alias; lý do được ghi trong `lexicon/cub/C/removed.csv`.

5. **Phạm vi xử lý D:** Mặc định D chỉ loại bỏ tiền tố thuộc tính của các lớp thuộc dataset đang xét. Không sử dụng `--d-strip-all` nếu chưa được nhóm thống nhất.

6. **Kiểm tra kết quả:** Sau khi chạy đủ ba dataset, kiểm tra coverage, audit và các trường hợp unmatched. Đặc biệt, `lexicon/{dataset}/D/tags.csv` để trống theo thiết kế. Cần đảm bảo bước xử lý tiếp theo có thể xử lý trường hợp này.

Trước khi bàn giao, chạy lại pre-push check và test suite theo README. Các trường hợp unmatched và những thay đổi trong `removed.csv`, `modified.csv` cần được rà soát trước khi kết luận kết quả đã hoàn tất.