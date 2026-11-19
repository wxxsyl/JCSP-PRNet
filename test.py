# import argparse
# import os
# import random
# import time
# import warnings
# import cv2
# import numpy as np
# import torch
# from thop import profile
# from Net import FusionNet
# from datasets import FusionDatasets
# from Evaluator.test_metric import evaluate

# warnings.filterwarnings("ignore")


# def parse_args():
#     parser = argparse.ArgumentParser(description='JCSSM Internal Ablation Testing')
    
#     # 1. 定义可选的数据集及其对应的固定路径映射
#     dataset_configs = {
#         "DP": "/hy-tmp/DronePersonTest/",
#         "DV": "/hy-tmp/DVTest/",
#         "TNO": "/hy-tmp/TNOTest/"
#     }
    
#     parser.add_argument('--dname', type=str, default="DP", choices=list(dataset_configs.keys()), help='数据集名称')
#     parser.add_argument('--data_root', type=str, default=None, help='测试数据集根目录路径（自动匹配）')

#     parser.add_argument('--pretrained_model', type=str, 
#                         default='ABL-JCSSM-fu3/DV_bs16_lr0.0001_ep50_ch32-64-96-128_d1111_w=5-5-0.5/epoch_50.pth', 
#                         help='预训练模型权重路径')
#     parser.add_argument('--save_results', type=str, default="Results/", help='融合图像保存根路径')
        
#     parser.add_argument('--method', type=str, default=None, help='实验名称')
#     # parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
#     parser.add_argument('--skip_profile', action='store_true', help='跳过 FLOPs 计算')
    
#     args = parser.parse_args()
    
#     if args.data_root is None:
#         args.data_root = dataset_configs[args.dname]
        
#     return args



# def test_model(args, device):

#     checkpoint = torch.load(args.pretrained_model, map_location='cpu')
#     state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
#     train_args = checkpoint.get('args', {})
#     if isinstance(train_args, argparse.Namespace):
#         train_args = vars(train_args)

#     print(train_args)
#     if args.method is None:
#         args.method = train_args.get(
#             'exp_name',
#             train_args.get('experiment_name', os.path.basename(os.path.dirname(os.path.abspath(args.pretrained_model))))
#         )

#     for key in ('in_ch', 'stem_ch', 'stage_channels', 'num_experts', 'depths'):
#         if key in train_args:
#             setattr(args, key, train_args[key])

#     model = FusionNet(
#         in_ch=args.in_ch,
#         stem_ch=args.stem_ch,
#         stage_channels=args.stage_channels,
#         num_experts=args.num_experts,
#         # depths=args.depths,
#     ).to(device)

#     model.load_state_dict(state_dict, strict=True)
#     model.eval()

#     if not args.skip_profile:
#         dummy_ir = torch.randn(1, args.in_ch, 128, 128, device=device)
#         dummy_vi = torch.randn(1, args.in_ch, 128, 128, device=device)
#         flops, _ = profile(model, inputs=(dummy_ir, dummy_vi), verbose=False)
#         print(f"FLOPs: {flops / 1e9:.4f} G | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.4f} M")

#     fu_root = os.path.join(args.save_results, args.dname)
#     os.makedirs(fu_root, exist_ok=True)

#     test_dataset = FusionDatasets(dataroot=args.data_root, is_train=False)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
#     visible_dir = os.path.join(args.data_root, 'test', 'vi')

#     total_inference_time = 0.0
#     num_images = 0

#     # 2. 推理与 3. 保存结果
#     for data in test_loader:
#         infrared, vis_gray = data[0], data[1]
#         name = data[-1][0]

#         infrared = infrared.to(device)
#         visible = vis_gray.to(device)

#         start_time = time.time()
#         with torch.inference_mode():
#             fused_image = model(infrared, visible)
#         total_inference_time += (time.time() - start_time)
#         num_images += 1

#         fused_gray = torch.clamp(fused_image, 0, 1).squeeze().detach().cpu().numpy()
#         fused_gray = (fused_gray * 255.0).astype(np.uint8)
#         save_path = os.path.join(fu_root, name)

#         if args.dname in ['DP', 'DV']:
#             img_bgr = cv2.imread(os.path.join(visible_dir, name))
#             img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
#             _, cr, cb = cv2.split(img_ycrcb)
#             fused_bgr = cv2.cvtColor(cv2.merge([fused_gray, cr, cb]), cv2.COLOR_YCrCb2BGR)
#             cv2.imwrite(save_path, fused_bgr)
#         else:
#             cv2.imwrite(save_path, fused_gray)

#     avg_time = total_inference_time / num_images
#     print(f"Inference Done. Avg Time: {avg_time:.4f}s | FPS: {1.0 / avg_time:.2f}")
#     print(f"Total Inference Time : {total_inference_time:.4f} s")
#     return fu_root


# def main():
#     args = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     fused_folder = test_model(args, device)
#     print(f"Evaluating fusion metrics...")
#     evaluate(test_folder=args.data_root, test_fuse_folder=fused_folder)



# if __name__ == "__main__":
#     main()







import argparse
import os
import random
import time
import warnings
import cv2
import numpy as np
import torch
from thop import profile
from Net import FusionNet
from datasets import FusionDatasets
from Evaluator.test_metric import evaluate

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='JCSSM Internal Ablation Testing')
    
    # 1. 定义可选的数据集及其对应的固定路径映射
    dataset_configs = {
        "DP": "/hy-tmp/DronePersonTest/",
        "DV": "/home/wanghan/HJJ-Code/Datasets/DVTest/",
        "TNO": "/hy-tmp/TNOTest/"
    }
    
    parser.add_argument('--dname', type=str, default="DV", choices=list(dataset_configs.keys()), help='数据集名称')
    parser.add_argument('--data_root', type=str, default=None, help='测试数据集根目录路径（自动匹配）')

    parser.add_argument('--pretrained_model', type=str, 
                        default='ABL-JCSSM-fu3/DV_bs16_lr0.0001_ep50_ch32-64-96-128_d1111_w=5-5-0.5/epoch_50.pth', 
                        help='预训练模型权重路径')
    parser.add_argument('--save_results', type=str, default="Results/", help='融合图像保存根路径')
        
    parser.add_argument('--method', type=str, default=None, help='实验名称')
    # parser.add_argument('--seed', type=int, default=42, help='全局随机种子')
    parser.add_argument('--skip_profile', action='store_true', help='跳过 FLOPs 计算')
    
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = dataset_configs[args.dname]
        
    return args



def test_model(args, device):

    checkpoint = torch.load(args.pretrained_model, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    train_args = checkpoint.get('args', {})
    if isinstance(train_args, argparse.Namespace):
        train_args = vars(train_args)

    print(train_args)
    if args.method is None:
        args.method = train_args.get(
            'exp_name',
            train_args.get('experiment_name', os.path.basename(os.path.dirname(os.path.abspath(args.pretrained_model))))
        )

    for key in ('in_ch', 'stem_ch', 'stage_channels', 'num_experts', 'depths'):
        if key in train_args:
            setattr(args, key, train_args[key])

    model = FusionNet(
        in_ch=args.in_ch,
        stem_ch=args.stem_ch,
        stage_channels=args.stage_channels,
        num_experts=args.num_experts,
        # depths=args.depths,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if not args.skip_profile:
        dummy_ir = torch.randn(1, args.in_ch, 128, 128, device=device)
        dummy_vi = torch.randn(1, args.in_ch, 128, 128, device=device)
        flops, _ = profile(model, inputs=(dummy_ir, dummy_vi), verbose=False)
        print(f"FLOPs: {flops / 1e9:.4f} G | Params: {sum(p.numel() for p in model.parameters()) / 1e6:.4f} M")

    # 2. 根据数据集配置保存路径（DP与DV分别设置 _gray 和 _rgb 文件夹）
    if args.dname in ['DP', 'DV']:
        gray_root = os.path.join(args.save_results, f"{args.dname}_gray")
        rgb_root = os.path.join(args.save_results, f"{args.dname}_rgb")
        os.makedirs(gray_root, exist_ok=True)
        os.makedirs(rgb_root, exist_ok=True)
    else:
        gray_root = os.path.join(args.save_results, args.dname)
        os.makedirs(gray_root, exist_ok=True)
        rgb_root = None

    test_dataset = FusionDatasets(dataroot=args.data_root, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    visible_dir = os.path.join(args.data_root, 'test', 'vi')

    total_inference_time = 0.0
    num_images = 0

    # 推理与保存结果
    for data in test_loader:
        infrared, vis_gray = data[0], data[1]
        name = data[-1][0]

        infrared = infrared.to(device)
        visible = vis_gray.to(device)

        start_time = time.time()
        with torch.inference_mode():
            fused_image = model(infrared, visible)
        total_inference_time += (time.time() - start_time)
        num_images += 1

        fused_gray = torch.clamp(fused_image, 0, 1).squeeze().detach().cpu().numpy()
        fused_gray = (fused_gray * 255.0).astype(np.uint8)

        # 1. 保存灰度图像到对应的目录
        save_path_gray = os.path.join(gray_root, name)
        cv2.imwrite(save_path_gray, fused_gray)

        # 2. 如果是 DP 或 DV，额外生成并保存 RGB 彩色融合图像到 _rgb 目录
        if args.dname in ['DP', 'DV']:
            save_path_rgb = os.path.join(rgb_root, name)
            img_bgr = cv2.imread(os.path.join(visible_dir, name))
            if img_bgr is not None:
                img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
                _, cr, cb = cv2.split(img_ycrcb)
                fused_bgr = cv2.cvtColor(cv2.merge([fused_gray, cr, cb]), cv2.COLOR_YCrCb2BGR)
                cv2.imwrite(save_path_rgb, fused_bgr)

    avg_time = total_inference_time / num_images
    print(f"Inference Done. Avg Time: {avg_time:.4f}s | FPS: {1.0 / avg_time:.2f}")
    print(f"Total Inference Time : {total_inference_time:.4f} s")
    
    # 返回灰度图路径用于后续指标评估（通常指标评估基于单通道灰度/亮度图）
    return gray_root


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fused_folder = test_model(args, device)
    print(f"Evaluating fusion metrics...")
    evaluate(test_folder=args.data_root, test_fuse_folder=fused_folder)


if __name__ == "__main__":
    main()