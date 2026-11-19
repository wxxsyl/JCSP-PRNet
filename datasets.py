import os
import cv2
import torch
from torch.utils.data import Dataset
from natsort import natsorted
import numpy as np


class FusionDatasets(Dataset):
    def __init__(self, dataroot, is_train=True):
        super().__init__()

        self.mode = 'train' if is_train else 'test'

        base_dir = dataroot
        self.vis_dir = os.path.join(base_dir, self.mode, 'vi')
        self.ir_dir = os.path.join(base_dir, self.mode, 'ir')

        self.image_list = natsorted(os.listdir(self.vis_dir))

    def __getitem__(self, index):
        name = self.image_list[index]

        ir_path = os.path.join(self.ir_dir, name)
        vi_path = os.path.join(self.vis_dir, name)

        ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.imread(vi_path)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        vi = img_ycrcb[:, :, 0]

        if ir is None or vi is None:
            raise RuntimeError(f"Image read failed: {name}")

        ir = ir.astype(np.float32) / 255.0
        vi = vi.astype(np.float32) / 255.0

        ir_tensor = torch.from_numpy(ir).unsqueeze(0)
        vi_tensor = torch.from_numpy(vi).unsqueeze(0)

        return ir_tensor, vi_tensor, name

    def __len__(self):
        return len(self.image_list)