# Báo cáo Lab 16 — Track 2: Thiết lập Môi trường AI trên Cloud (AWS)

**Sinh viên:** Nguyễn Thành Toàn — `2A202600633`
**Phương án:** CPU Instance + LightGBM (dự phòng khi không có quota GPU)
**Region:** `us-east-1`
**Ngày:** 2026-06-18

---

## 1. Tóm tắt

Do tài khoản AWS mới không có quota GPU (mặc định 0 vCPU cho dòng G/VT, yêu cầu
tăng quota bị trì hoãn/từ chối), em chuyển sang **phương án CPU**: train mô hình
**LightGBM** phát hiện gian lận thẻ tín dụng trên instance `r5.2xlarge`.

Pipeline đầy đủ: **Terraform IaC → EC2 (Bastion + CPU node private) → NAT → ALB →
Training → Inference → kiểm tra chi phí**.

---

## 2. Kiến trúc hạ tầng (27 resources)

```
Internet
   │
   ├─ ALB :80 ───────────► (target) CPU Node :8000     [public → private]
   │
   └─ Bastion :22 (public subnet, IP công khai)
            │ SSH (ProxyJump)
            ▼
        CPU Node  r5.2xlarge  (private subnet, KHÔNG IP công khai)
            │ outbound
            ▼
        NAT Gateway ───► Internet  (tải apt / pip / dataset)
```

| Thành phần | Vai trò |
|---|---|
| VPC + 2 public + 2 private subnet | Mạng cô lập, đa AZ |
| Bastion (`t3.micro`, public) | Cửa SSH duy nhất, chắn truy cập thẳng node |
| CPU Node (`r5.2xlarge`, private) | Train LightGBM, ẩn khỏi internet |
| NAT Gateway | Cho node private ra internet, không nhận inbound |
| ALB | Điểm vào HTTP công khai, health check `/health` |
| Security Groups | SSH chỉ từ Bastion; `:8000` chỉ từ ALB |
| IAM Role + Instance Profile | Quyền cho node, không nhúng credentials |

**Ghi chú AMI:** Terraform tự resolve AMI bằng `data "aws_ami"` (Ubuntu 22.04 cho
Bastion; Deep Learning Ubuntu 22.04 cho node) — không hardcode AMI ID, tự lấy bản
mới nhất theo region.

---

## 3. Dataset

- **Nguồn:** Kaggle `mlg-ulb/creditcardfraud` (Credit Card Fraud Detection)
- **Kích thước:** 284,807 giao dịch, 30 features (V1–V28 ẩn danh PCA + Time + Amount)
- **Mất cân bằng:** chỉ ~0.17% là gian lận (fraud)

---

## 4. Kết quả Benchmark trên `r5.2xlarge`

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.071 s |
| Thời gian training | 1.527 s |
| Best iteration | 68 |
| AUC-ROC | 0.9103 |
| Accuracy | 0.9735 |
| F1-Score | 0.0992 |
| Precision | 0.0527 |
| Recall | 0.8469 |
| Inference latency (1 row) | 0.374 ms |
| Inference throughput (1000 rows) | ~837,762 rows/s |

> Số liệu đầy đủ trong [benchmark_result.json](benchmark_result.json). Mã nguồn:
> [benchmark.py](benchmark.py).

---

## 5. Phân tích kết quả

- **AUC-ROC 0.910 + Recall 0.847 cao** → mô hình phân biệt tốt và **bắt được
  84.7% giao dịch gian lận**.
- **Precision 0.053 / F1 0.099 thấp là chủ đích**, không phải lỗi: tham số
  `scale_pos_weight ≈ 578` ép mô hình ưu tiên lớp thiểu số (fraud chỉ 0.17%) →
  nhiều cảnh báo nhầm nhưng ít bỏ sót. Đây là đánh đổi hợp lý trong chống gian lận:
  **thà báo nhầm còn hơn sót fraud**.
- **Accuracy 0.974 gần như vô nghĩa** ở dữ liệu mất cân bằng — chỉ cần đoán "tất cả
  hợp lệ" đã đạt ~99.8%. Phải đánh giá bằng **AUC + Recall**, không nhìn accuracy.
- **Hiệu năng:** training 1.5s, inference 0.374 ms/dòng — CPU cao cấp thừa sức cho
  bài toán dữ liệu bảng (tabular).

---

## 6. So sánh CPU vs GPU

| | CPU (`r5.2xlarge`) | GPU (`g4dn.xlarge`) |
|---|---|---|
| Chi phí | ~$0.504/giờ | ~$0.526/giờ |
| Cần quota đặc biệt | **Không** | Có (account mới thường bị từ chối) |
| Hợp với | Gradient boosting / tabular ML | Deep learning / LLM |

**Kết luận:** LightGBM (gradient boosting trên dữ liệu bảng) **không cần GPU**.
CPU cao cấp chạy nhanh, chi phí tương đương, lại không vướng quota — phù hợp tài
khoản mới. GPU chỉ thực sự lợi cho deep learning / LLM.

---

## 7. Chi phí (us-east-1)

| Dịch vụ | Loại | Chi phí/giờ |
|---|---|---|
| EC2 — CPU Node | `r5.2xlarge` | ~$0.504 |
| EC2 — Bastion | `t3.micro` | ~$0.010 |
| NAT Gateway | (mỗi AZ) | ~$0.045 + data |
| ALB | Application LB | ~$0.008 |
| **Tổng** | | **~$0.57/giờ** |

> Đã chạy `terraform destroy` sau khi benchmark xong để tránh phát sinh chi phí.
> NAT Gateway + EIP tính tiền kể cả khi rảnh → destroy là bắt buộc.

---

## 8. Vấn đề gặp & cách xử lý

| Vấn đề | Nguyên nhân | Cách xử lý |
|---|---|---|
| SSH `Permission denied (publickey)` | Chạy lại `ssh-keygen` sau `apply` → key trên instance ≠ key local | Dựng lại hạ tầng, **không** regenerate key |
| ProxyJump vẫn denied ở Bastion | Chặng jump không tự dùng `-i lab-key` | Thêm `-o ProxyCommand="ssh -i lab-key -W %h:%p ..."` + `IdentitiesOnly=yes` |
| `ec2-user` không vào được | README giả định Amazon Linux, infra thực là Ubuntu | Dùng user `ubuntu`, package manager `apt` (không `dnf`) |
| `pip install` bị chặn (PEP 668) | Ubuntu mới chặn cài global | Dùng `python3 -m venv` |

---

## 9. Deliverables

1. ✅ Screenshot terminal chạy `python3 benchmark.py`
2. ✅ [benchmark_result.json](benchmark_result.json)
3. ✅ Screenshot AWS Billing (EC2 + NAT Gateway)
4. ✅ Mã nguồn `terraform/` (đã chỉnh `r5.2xlarge`)
5. ✅ Báo cáo này
