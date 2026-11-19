import os
import cv2
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from .Evaluator import Evaluator


# ===================== 图像读取 =====================
def image_read_cv2(path, mode='GRAY'):
    img = cv2.imread(path)

    if mode == 'RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    elif mode == 'GRAY':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif mode == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    else:
        raise ValueError("mode error")


# ===================== 单张计算 =====================
def process_one(data):
    vi, ir, fi = data

    return np.array([
        Evaluator.EN(fi),
        Evaluator.SD(fi),
        Evaluator.SF(fi),
        Evaluator.AG(fi),
        Evaluator.MI(fi, ir, vi),
        Evaluator.SCD(fi, ir, vi),
        Evaluator.VIFF(fi, ir, vi),
        Evaluator.Qabf(fi, ir, vi),
        Evaluator.SSIM(fi, ir, vi)
    ], dtype=np.float32)


# ===================== 主评估函数 =====================
def evaluate(test_folder, test_fuse_folder, decimals=4, num_workers=None):
    """
    高性能CPU评估函数
    """

    start_time = time.time()

    src1_path = os.path.join(test_folder, "test/vi")
    src2_path = os.path.join(test_folder, "test/ir")

    path1 = sorted(os.listdir(src1_path))
    path2 = sorted(os.listdir(src2_path))

    assert len(path1) == len(path2), "VI 和 IR 数量不一致"

    num_imgs = len(path1)

    # ===================== 1. 一次性读取所有图像（关键优化） =====================
    print("Loading images into memory...")
    vi_imgs = [image_read_cv2(os.path.join(src1_path, n), 'GRAY') for n in path1]
    ir_imgs = [image_read_cv2(os.path.join(src2_path, n), 'GRAY') for n in path2]
    fi_imgs = [image_read_cv2(os.path.join(test_fuse_folder, n), 'GRAY') for n in path1]

    data_list = list(zip(vi_imgs, ir_imgs, fi_imgs))

    # ===================== 2. 多进程 =====================
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} CPU cores...")

    with Pool(num_workers) as pool:
        results = pool.map(process_one, data_list)

    # ===================== 3. 汇总 =====================
    metric_result = np.mean(results, axis=0)

    # ===================== 4. 保留小数 =====================
    metric_result = np.round(metric_result, decimals)
    metric_names = ["EN", "SD", "SF", "AG", "MI", "SCD", "VIF", "Qabf", "SSIM"]

    end_time = time.time()

    print("\n=== Fusion Metrics ===")
    for name, val in zip(metric_names, metric_result):
        print(f"{name:<5}: {val:.4f}")

    print(f"\nTotal Time: {end_time - start_time:.2f}s")
