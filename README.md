# T-Graph

**Official implementation of**  
**T-Graph: Enhancing Sparse-view Camera Pose Estimation by Pairwise Translation Graph**

📄 Paper (ISPRS Journal of Photogrammetry and Remote Sensing):  
🔗 https://doi.org/10.1016/j.isprsjprs.2025.08.031

---

## 🔽 Pretrained Weights

Pretrained model weights are available at:  
🔗 https://drive.google.com/drive/folders/1l-X36atuvW3WDNhOxZ5a1oe5IdrvnWc1?usp=drive_link

---

## 🚀 Training

All training scripts are based on **PyTorch Distributed Data Parallel (DDP)**.

### 1️⃣ Train RelPose++ with **Pairwise Translation Graph (pair-t)**

```bash
torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  --nproc_per_node=1 \
  trainer_ddp_pairT.py \
  --batch_size=22 \
  --num_images=8 \
  --random_num_images=true \
  --gpu_ids=0 \
  --lr=1e-5 \
  --normalize_cameras \
  --use_amp
````

---

### 2️⃣ Train RelPose++ with **Relative Translation Graph (relative-t)**

```bash
torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  --nproc_per_node=1 \
  trainer_ddp_relativeT.py \
  --batch_size=22 \
  --num_images=8 \
  --random_num_images=true \
  --gpu_ids=0 \
  --lr=1e-5 \
  --normalize_cameras \
  --use_amp
```


## 🙏 Acknowledgements

We sincerely thank the authors of **RelPose++** for their excellent open-source contribution.
Part of this work is built upon RelPose++, and we reused and adapted several components of their codebase.

For environment setup, data preprocessing, and general training instructions, please refer to:
🔗 [https://github.com/amyxlase/relpose-plus-plus](https://github.com/amyxlase/relpose-plus-plus)


## 📌 Notes

- Additional code and utilities will be released progressively.
- If you find this repository useful, please consider citing our paper.