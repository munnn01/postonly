# Post-only: hậu xử lý video cho nhận dạng

Baseline độc lập: **video gốc → H.264/H.265 thật → postprocessor → analyzer đóng băng**.

Mục tiêu là đo tăng Top-1 mà không thay bitstream. **Chưa có checkpoint train trên dữ liệu thật hoặc kết quả chứng minh BD-rate < −10%.** Không cần rate proxy: codec ở trước module học. Bitrate không phụ thuộc trọng số postprocessor, vì vậy trainer không có rate loss giả tạo.

## Phạm vi phiên bản này

- Postprocessor residual Conv3D nhỏ, có QP conditioning, khởi tạo identity.
- Analyzer torchvision pretrained, đóng băng tham số nhưng giữ gradient theo video đầu vào.
- Codec FFmpeg thật, tính bpp từ byte elementary stream / (số frame × H × W).
- Split class-balanced chính xác 2800 train / 700 val nếu dữ liệu đủ, manifest cố định và không thay video lỗi bằng mẫu từ split khác.
- Train riêng từng codec, QP 30/35/40/45; CE + MSE tùy trọng số.
- Chọn checkpoint theo validation BD-rate Top-1 hợp lệ; trước khi có BD-rate hợp lệ dùng accuracy làm fallback, được ghi trong history.
- Đánh giá paired anchor/postonly trên cùng decoded clip và cùng bpp, PCHIP log-rate với miền Top-1 giao nhau; paired bootstrap theo video.
- Checkpoint `weights_only=True`, output phải rỗng để tránh ghi đè dữ liệu cũ.

**Chưa triển khai:** preprocessor/rate-weight sweep, KD, adaptive rate controller, audit rate proxy, resume, cache codec và multi-GPU. Đây là giai đoạn post-only của roadmap, không phải toàn bộ hệ thống sandwich mới. Code standalone viết mới; không sao chép repo preprocessing gốc hoặc phụ thuộc đường dẫn repo đó.

## Cài đặt

Python 3.10+, PyTorch/torchvision tương thích; FFmpeg cần `libx264` và `libx265`.

```bash
python -m pip install -e ".[test]"
```

Cài PyTorch phù hợp GPU trước nếu cần. Analyzer tải trọng số torchvision lần đầu nếu chưa có cache; dữ liệu video không được gửi lên dịch vụ ngoài.

## Dữ liệu

```text
videos/
  archery/
    clip001.mp4
  playing guitar/
    clip002.mp4
```

Tên thư mục lớp phải khớp label trong `FrozenAnalyzer.categories` (Kinetics-400 với `r3d_18`). `--data-root` trỏ tới một pool để tạo split, không phải thư mục chứa sẵn train/val lồng nhau. Chuẩn bị pool không trùng nội dung và kiểm tra quyền sử dụng dữ liệu trước khi chạy. Split manifest lưu đường dẫn tương đối; không commit manifest/dữ liệu riêng tư.

Video lỗi sẽ báo lỗi, không âm thầm đổi mẫu. Video ngắn lặp frame theo quy tắc dataset. Đây là mã hóa các clip ngắn được lấy mẫu, không phải bitrate toàn video gốc. FPS encode, stride, frame size và preset là một phần protocol.

## Smoke test thật, không tải trọng số

```bash
python scripts/smoke.py --ffmpeg /path/to/ffmpeg --codec h264
python scripts/smoke.py --ffmpeg /path/to/ffmpeg --codec h265
python -m postonly.cli smoke --ffmpeg /path/to/ffmpeg --output runs/smoke
```

`scripts/smoke.py` độc lập kiểm tra FFmpeg và convolution đơn giản. Lệnh CLI kiểm tra chính model/codec của package với classifier tổng hợp. Cả hai không đo Top-1 thực hoặc BD-rate thực nghiệm.

## Train riêng từng codec: 2800/700

```bash
python -m postonly.cli train --data-root /path/to/videos --output runs/h264_seed42 --codec h264 --limit-train 2800 --limit-val 700 --qps 30 35 40 45 --epochs 10 --device cuda --ffmpeg /path/to/ffmpeg
python -m postonly.cli train --data-root /path/to/videos --output runs/h265_seed42 --codec h265 --limit-train 2800 --limit-val 700 --qps 30 35 40 45 --epochs 10 --device cuda --ffmpeg /path/to/ffmpeg
```

Giữ nguyên dữ liệu, seed, danh sách QP, temporal/spatial protocol khi so codec và ablation. Seed cố định không bảo đảm bitwise deterministic trên mọi GPU/version. Mỗi batch train lấy một QP; validation chạy đủ mọi QP. Chưa có cache nên mã hóa lại mỗi epoch và validation có thể chậm.

Đường dẫn môi trường Windows đã dùng kiểm tra (thay bằng đường dẫn của bạn):

```powershell
& 'D:\STUDY\AI\envs\ten_env\python.exe' -B -m postonly.cli train --data-root 'D:\datasets\videos' --output runs/h264_seed42 --codec h264 --device cpu --ffmpeg 'D:\STUDY\AI\envs\ten_env\Library\bin\ffmpeg.exe'
```

Chạy tại thư mục clone repo. GPU khuyến nghị cho dữ liệu thực; `--device cpu` chỉ phù hợp smoke hoặc thử nhỏ. Không có cam kết thời gian train.

Outputs: `best.pt`, `last.pt`, `history.json`, `split_manifest.json`, `per_video_metrics.csv`, `bd_rate.json`. Checkpoint không hỗ trợ resume optimizer trong phiên bản này. Các file trong `runs/` bị Git ignore.

## Đánh giá lại validation hoặc test riêng

Validation đúng manifest của checkpoint:

```bash
python -m postonly.cli evaluate --data-root /path/to/videos --checkpoint runs/h264_seed42/best.pt --manifest runs/h264_seed42/split_manifest.json --output runs/h264_validation --bootstrap 10000 --ffmpeg /path/to/ffmpeg
```

Test riêng (không truyền manifest, chạy toàn bộ test root):

```bash
python -m postonly.cli evaluate --data-root /path/to/independent_test --checkpoint runs/h264_seed42/best.pt --output runs/h264_test --bootstrap 10000 --ffmpeg /path/to/ffmpeg
```

Codec/QP/analyzer/shape/preset được lấy từ checkpoint. Bạn phải xác nhận test độc lập trước khi tuning. Không dùng lại val đã chọn checkpoint để tuyên bố kết quả test. 700 val nhỏ; paired bootstrap không thay thế test độc lập. Số bootstrap mặc định 1000 phục vụ phát triển; dùng 10000 cho báo cáo, kèm số bootstrap hợp lệ/không hợp lệ. Không tự gắn `Reportable: True`.

## Đọc BD-rate

- Âm là tiết kiệm bitrate tại cùng Top-1, **không** có nghĩa postprocessor giảm byte ở cùng QP.
- Anchor/postonly tại mỗi video/QP luôn dùng cùng rate; postprocessor chỉ làm thay đổi quality của đường cong.
- Miền Top-1 giao nhau có thể đổi giữa checkpoint. Đường cong phẳng hoặc không có miền giao nhau cho kết quả undefined, không đổi thành 0.
- Khi tăng accuracy đầu mút làm đổi miền tích phân, scalar BD-rate có thể xấu hơn dù một điểm tốt lên. Báo miền và đường cong, không chọn/bỏ QP theo nhãn test.
- H.264 và H.265 phải báo riêng. Không gộp hai codec vào một đường cong.
- Chi phí model/compute không nằm trong bpp; giả định postprocessor được chia sẻ sẵn ở decoder, không truyền tham số từng video.

## Kiểm thử

```bash
python -m pytest -q
```

Tests dùng dữ liệu/classifier tổng hợp, không tải dataset hoặc trọng số. Smoke FFmpeg kiểm tra hai encoder thật. Trainer integration test dùng codec và analyzer giả có kiểm soát để kiểm tra loop/checkpoint; không được hiểu là train đã chạy với analyzer pretrained và dữ liệu thật.

## Roadmap tiếp theo

1. Audit kết quả H.265 QP30 trong hệ thống sandwich gốc và khóa protocol.
2. Chạy post-only riêng hai codec bằng repo này, so anchor/post-only.
3. Chỉ sau baseline đáng tin mới thêm preprocessor gần identity và so đủ anchor/post-only/pre-only/sandwich.
4. Sweep rate weight nhỏ chỉ có ý nghĩa khi có preprocessor và rate proxy được kiểm chứng.
5. Thêm clean-teacher KD sau, không thay analyzer đóng băng thành mạng đang học.

Không có tuyên bố license cho mã bên thứ ba/weights/dataset; người dùng cần tuân thủ giấy phép của PyTorch, torchvision, FFmpeg và dữ liệu mình sử dụng.
