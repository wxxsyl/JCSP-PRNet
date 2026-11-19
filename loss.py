
import torch
import torch.nn as nn
import torch.nn.functional as F
# import kornia
from Loss_ssim import ssim


# class SobelGradient(nn.Module):
#     def __init__(self, channels=1):
#         super().__init__()
#         channels = int(channels)
#         sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)
#         sobel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)
#         self.register_buffer('sobel_x', sobel_x)
#         self.register_buffer('sobel_y', sobel_y)
#         self.channels = channels

#     def forward(self, x):
#         padded_x = F.pad(x, (1, 1, 1, 1), mode='replicate')
#         kernel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
#         kernel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)
#         g1 = F.conv2d(padded_x, kernel_x, groups=self.channels)
#         g2 = F.conv2d(padded_x, kernel_y, groups=self.channels)
#         return torch.abs(g1) + torch.abs(g2)

# class FusionLoss(nn.Module):
#     def __init__(self, channels=1, window_size=11, eps=1e-6):
#         super().__init__()
#         self.channels = int(channels)
#         self.window_size = window_size
#         self.eps = eps
#         self.sobel = SobelGradient(channels=self.channels)

#     def intensity_loss(self, fused, infrared, visible):
#         target = torch.max(infrared, visible)
#         return F.l1_loss(fused, target)

#     def gradient_loss(self, fused, infrared, visible):
#         gradient_fused = self.sobel(fused)
#         gradient_ir = self.sobel(infrared)
#         gradient_vis = self.sobel(visible)
#         target_gradient = torch.max(gradient_ir, gradient_vis)
#         return F.l1_loss(gradient_fused, target_gradient)

#     def structural_consistency_loss(self, fused, infrared, visible):
#         gradient_vis = self.sobel(visible)
#         gradient_ir = self.sobel(infrared)
#         mean_gradient_vis = torch.mean(gradient_vis)
#         mean_gradient_ir = torch.mean(gradient_ir)
#         denominator = mean_gradient_vis + mean_gradient_ir + self.eps
#         weight_vis = mean_gradient_vis / denominator
#         weight_ir = mean_gradient_ir / denominator
#         ssim_vis = ssim(visible, fused)
#         ssim_ir = ssim(infrared, fused)
#         weighted_ssim = weight_vis * ssim_vis + weight_ir * ssim_ir
#         return 1.0 - weighted_ssim

#     def forward(self, fused, infrared, visible):
#         loss_int = self.intensity_loss(fused, infrared, visible)
#         loss_grad = self.gradient_loss(fused, infrared, visible)
#         loss_ssim = self.structural_consistency_loss(fused, infrared, visible)
#         return loss_int, loss_grad, loss_ssim


# class FusionLoss(nn.Module):
#     def __init__(self, channels=1, window_size=11):
#         super(FusionLoss, self).__init__()

#         self.sobel = SobelGradient(channels=channels)

#         # Kornia SSIM
#         self.ssim = kornia.losses.SSIMLoss(
#             window_size=window_size,
#             reduction='mean'
#         )

#     def int_loss(self, fused_image, target_image):
#         """强度损失"""
#         return F.l1_loss(fused_image, target_image)

#     def grad_loss(self, fused_image, vis_image, inf_image):
#         """梯度损失"""
#         grad_fused = self.sobel(fused_image)
#         grad_vis = self.sobel(vis_image)
#         grad_inf = self.sobel(inf_image)

#         target_grad = torch.max(grad_vis, grad_inf)

#         return F.l1_loss(grad_fused, target_grad)

#     def forward(self, If, Ir, Iv):
#         """
#         :param If: 融合图像 (Fused Image)
#         :param Ir: 红外图像 (Infrared Image)
#         :param Iv: 可见光图像 (Visible Image)
#         :return: L_int, L_grad, L_ssim
#         """
#         If = torch.clamp(If, 0.0, 1.0)

#         # 1. 构造目标图
#         target_img = torch.max(Ir, Iv)

#         # 2. 计算三项损失
#         L_int = self.int_loss(If, target_img)
#         L_grad = self.grad_loss(If, Iv, Ir)
#         L_ssim = self.ssim(If, target_img)

#         # 3. 返回三项 Loss
#         return L_int, L_grad, L_ssim







class SobelGradient(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        channels = int(channels)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        sobel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]], dtype=torch.float32).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.channels = channels

    def forward(self, x):
        padded_x = F.pad(x, (1, 1, 1, 1), mode='replicate')
        kernel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        kernel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)
        g1 = F.conv2d(padded_x, kernel_x, groups=self.channels)
        g2 = F.conv2d(padded_x, kernel_y, groups=self.channels)
        return torch.abs(g1) + torch.abs(g2)


# class FusionLoss(nn.Module):
#     """
#     输入：同形状 [B, C, H, W]，源图范围为 [0, 1]。
#     返回：未加权的强度、梯度、结构损失。
#     """

#     def __init__(self, channels=1):
#         super().__init__()
#         self.channels = channels
#         self.sobel = SobelGradient(channels)

#     def forward(self, fused, infrared, visible):
#         if (
#             fused.ndim != 4
#             or fused.shape != infrared.shape
#             or fused.shape != visible.shape
#             or fused.shape[1] != self.channels
#         ):
#             raise ValueError(
#                 "Inputs must share [B, channels, H, W] shape"
#             )

#         # FP32 计算损失；不截断 fused，保留越界预测的梯度。
#         fused = fused.float()
#         infrared = infrared.float()
#         visible = visible.float()

#         target_intensity = torch.maximum(infrared, visible)
#         loss_int = F.l1_loss(fused, target_intensity)

#         gradient_fused = self.sobel(fused)
#         target_gradient = torch.maximum(
#             self.sobel(infrared),
#             self.sobel(visible),
#         )
#         loss_grad = F.l1_loss(gradient_fused, target_gradient)


#         loss_ssim = 1 - 0.5 * (
#             ssim(fused, infrared)
#             + ssim(fused, visible)
#         )

#         return loss_int, loss_grad, loss_ssim






class FusionLoss(nn.Module):
    def __init__(self, channels=1, ssim_mode="mix", alpha=0.5):
        super().__init__()

        if ssim_mode not in ["mix", "avg", "max"]:
            raise ValueError(f"ssim_mode must be one of ['mix', 'avg', 'max'], but got {ssim_mode}")

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

        self.ssim_mode = ssim_mode
        self.alpha = alpha
        self.sobel = SobelGradient(channels)

    def forward(self, fused, infrared, visible):
        # FP32 计算损失；不截断 fused，保留越界预测的梯度。
        with torch.autocast(device_type=fused.device.type, enabled=False):
            fused = fused.float()
            infrared = infrared.float()
            visible = visible.float()

            # ==================================================
            # 1. 强度损失
            # ==================================================
            target_max = torch.maximum(infrared, visible)
            loss_int = F.l1_loss(fused, target_max)

            # ==================================================
            # 2. 梯度损失
            # ==================================================
            gradient_fused = self.sobel(fused)
            target_gradient = torch.maximum(self.sobel(infrared), self.sobel(visible))
            loss_grad = F.l1_loss(gradient_fused, target_gradient)

            # ==================================================
            # 3. SSIM 损失
            # ==================================================
            # 与红外、可见光分别计算 SSIM，然后取平均
            loss_ssim_avg = 1.0 - 0.5 * (ssim(fused, infrared) + ssim(fused, visible))

            # 与最大值强度图计算 SSIM
            loss_ssim_max = 1.0 - ssim(fused, target_max)

            # --------------------------
            # SSIM 模式选择
            # --------------------------
            if self.ssim_mode == "mix":
                loss_ssim = (1.0 - self.alpha) * loss_ssim_avg + self.alpha * loss_ssim_max
            elif self.ssim_mode == "avg":
                loss_ssim = loss_ssim_avg
            elif self.ssim_mode == "max":
                loss_ssim = loss_ssim_max

        return loss_int, loss_grad, loss_ssim