import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from Net import FusionNet
from datasets import FusionDatasets
from loss import FusionLoss


def parse_args():
    parser = argparse.ArgumentParser(description='JCSSM Internal Ablation Training')
    
    # ------------------ 数据与路径参数 ------------------
    parser.add_argument('--root_path', type=str, default='/hy-tmp/CropDV_Patches725-patch8/', help='训练数据集根目录')
    parser.add_argument('--output', type=str, default='./ABL-JCSSM-fu3', help='输出根目录')
    parser.add_argument('--dataset_name', type=str, default='DV', help='数据集名称')
    parser.add_argument('--img_size', type=int, default=128, help='输入图像尺寸')
    
    # ------------------ 训练超参数 ------------------
    parser.add_argument('--epochs', type=int, default=50, help='训练总轮数')
    parser.add_argument('--save_freq', type=int, default=10, help='权重保存间隔轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='每张卡的 batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    
    # ------------------ 损失函数权重参数 ------------------
    parser.add_argument('--loss_int_weight', type=float, default=5.0, help='强度损失权重')
    parser.add_argument('--loss_grad_weight', type=float, default=5.0, help='梯度损失权重')
    parser.add_argument('--loss_ssim_weight', type=float, default=1.0, help='SSIM损失权重')
    parser.add_argument('--ssim_mode', type=str, default='avg', help='SSIM计算模式')

    # ------------------ 网络结构参数 ------------------
    parser.add_argument('--in_ch', type=int, default=1, help='输入通道数 (IR/VIS)')
    parser.add_argument('--stem_ch', type=int, default=32, help='Stem层输出通道数')
    parser.add_argument('--stage_channels', type=int, nargs='+', default=[32, 64, 96, 128], help='编码器各级通道数')
    parser.add_argument('--num_experts', type=int, default=3, help='路由专家数量')
    # parser.add_argument('--depths', type=int, nargs='+', default=[1, 1, 2, 2], help='交叉融合模块深度')

    # ------------------ 其他控制参数 ------------------
    parser.add_argument('--n_gpu', type=int, default=1, help='总 GPU 数量')
    parser.add_argument('--deterministic', type=int, default=1, help='是否使用确定性训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_resume', action='store_true', help='不进行断点续训')
    
    return parser.parse_args()


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_train_loader(args):
    dataset = FusionDatasets(dataroot=args.root_path, is_train=True)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = dict(
        dataset=dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True, generator=generator,
    )
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = False
        loader_kwargs['prefetch_factor'] = 4
    return DataLoader(**loader_kwargs)


def restore_checkpoint(model, optimizer, resume_path, args):
    start_epoch = 0
    best_loss = float('inf')

    if not resume_path or not os.path.isfile(resume_path):
        return start_epoch, best_loss

    print(f'==> Resume from: {resume_path}')
    ckpt = torch.load(resume_path, map_location='cpu')

    checkpoint_epoch = int(ckpt['epoch'])
    if checkpoint_epoch >= args.epochs:
        print(f'==> Training already completed: {checkpoint_epoch}/{args.epochs}')
        return checkpoint_epoch, float(ckpt.get('best_loss', float('inf')))

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    start_epoch = checkpoint_epoch
    best_loss = float(ckpt.get('best_loss', float('inf')))
    print(f'==> Resumed successfully! Next epoch: {start_epoch + 1}, Best loss: {best_loss:.6f}')
    return start_epoch, best_loss


def train_model(args, net, snapshot_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = build_train_loader(args)

    log_path = os.path.join(snapshot_path, 'train_log.csv')
    latest_path = os.path.join(snapshot_path, 'latest_model.pth')
    best_path = os.path.join(snapshot_path, 'best_model.pth')
    resume_path = latest_path if not args.no_resume else None

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FusionLoss(ssim_mode=args.ssim_mode, alpha=0.5).to(device)
    
    start_epoch, best_loss = restore_checkpoint(net, optimizer, resume_path, args)

    # 如果已经训练完成了，直接返回
    if start_epoch >= args.epochs:
        print(f"==> 实验 {snapshot_path} 已经全部完成，跳过训练。")
        return

    if start_epoch == 0:
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['Epoch', 'LR', 'AvgTotalLoss', 'AvgIntLoss', 'AvgGradLoss', 'AvgSSIMLoss'])
    elif not os.path.isfile(log_path):
        raise FileNotFoundError(f'Resuming but missing train_log.csv: {log_path}')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        net.train()
        if loader.generator is not None:
            loader.generator.manual_seed(args.seed + epoch)
            
        totals = np.zeros(4, dtype=np.float64)
        valid_batches = 0
        pbar = tqdm(loader, desc=f'Epoch [{epoch}/{args.epochs}]', ncols=140)

        for batch in pbar:
            ir, vi = batch[:2]
            ir = ir.to(device, non_blocking=True)
            vi = vi.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            output = net(ir, vi)
            fused = output[0] if isinstance(output, (tuple, list)) else output
            
            loss_int, loss_grad, loss_ssim = criterion(fused, ir, vi)
            loss = args.loss_int_weight * loss_int + args.loss_grad_weight * loss_grad + args.loss_ssim_weight * loss_ssim
            
            if not torch.isfinite(loss):
                print(f'[WARNING] Non-finite loss in epoch {epoch}, skipping batch')
                continue
                
            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.grad_clip)
                
            optimizer.step()
            
            totals += [loss.item(), loss_int.item(), loss_grad.item(), loss_ssim.item()]
            valid_batches += 1
            pbar.set_postfix(LR=f'{optimizer.param_groups[0]["lr"]:.2e}', Loss=f'{loss.item():.4f}')

        if valid_batches == 0:
            raise RuntimeError('No valid batch in epoch')
            
        averages = totals / valid_batches
        current_lr = optimizer.param_groups[0]['lr']

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, current_lr, *averages.tolist()])
            
        is_best = averages[0] < best_loss
        if is_best:
            best_loss = float(averages[0])

        state = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'args': vars(args),
        }
        torch.save(state, latest_path)
        if is_best:
            torch.save(state, best_path)
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            epoch_path = os.path.join(snapshot_path, f'epoch_{epoch}.pth')
            torch.save(state, epoch_path)
            print(f'Saved: {epoch_path}')

        print(f'[Epoch {epoch:03d}] LR={current_lr:.3e}, Total={averages[0]:.4f}, '
              f'Int={averages[1]:.4f}, Grad={averages[2]:.4f}, SSIM={averages[3]:.4f}')


if __name__ == "__main__":
    args = parse_args()

    # 1. 确定性与硬件加速设置
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 定义 7 组多参数实验配置列表
    experiments = [
        # { 'batch_size': 16, 'weights': (5.0, 5.0, 0.5), 'depths': [1, 1, 2, 2] },
        # { 'batch_size': 24, 'weights': (5.0, 5.0, 0.5), 'depths': [1, 1, 2, 2] },
        { 'batch_size': 16, 'weights': (5.0, 5.0, 0.5), 'depths': [1, 1, 1, 1] },
        # { 'batch_size': 16, 'weights': (5.0, 5.0, 0.5), 'depths': [2, 2, 2, 2] },
        # { 'batch_size': 16, 'weights': (5.0, 5.0, 1.0), 'depths': [1, 1, 1, 1] },
        # { 'batch_size': 16, 'weights': (5.0, 5.0, 1.0), 'depths': [2, 2, 2, 2] },
        # { 'batch_size': 24, 'weights': (5.0, 5.0, 1.0), 'depths': [1, 1, 2, 2] },
    ]

    # 2. 循环依次执行 7 组实验
    for idx, exp_cfg in enumerate(experiments, 1):
        # 更新当前实验参数
        args.batch_size = exp_cfg['batch_size']
        args.loss_int_weight, args.loss_grad_weight, args.loss_ssim_weight = exp_cfg['weights']
        args.depths = exp_cfg['depths']

        # 动态生成实验独立目录
        ch_string = '-'.join(str(c) for c in args.stage_channels)
        depth_string = ''.join(str(d) for d in args.depths)
        weight_string = f'{args.loss_int_weight:g}-{args.loss_grad_weight:g}-{args.loss_ssim_weight:g}'
        
        args.exp = f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr:g}_ep{args.epochs}_ch{ch_string}_d{depth_string}_w={weight_string}"
        snapshot_path = os.path.join(args.output, args.exp)
        os.makedirs(snapshot_path, exist_ok=True)

        # 检查是否已经完整跑完当前实验
        latest_path = os.path.join(snapshot_path, 'latest_model.pth')
        skip_current = False
        if os.path.isfile(latest_path):
            ckpt_temp = torch.load(latest_path, map_location='cpu')
            if int(ckpt_temp.get('epoch', 0)) >= args.epochs:
                print(f"\n[跳过] 第 {idx}/7 组实验 ({args.exp}) 已经全部完成 ({ckpt_temp.get('epoch')}/{args.epochs} epochs)。")
                skip_current = True

        if skip_current:
            continue

        print(f"\n==========================================")
        print(f"开始/断点续训 第 {idx}/7 组实验:")
        print(f"  - 目录:       {snapshot_path}")
        print(f"  - batch_size: {exp_cfg['batch_size']}")
        print(f"  - 损失权重:   {exp_cfg['weights']}")
        print(f"  - 深度(depths): {exp_cfg['depths']}")
        print(f"==========================================")

        # 实例化模型
        net = FusionNet(
            in_ch=args.in_ch,
            stem_ch=args.stem_ch,
            stage_channels=args.stage_channels,
            num_experts=args.num_experts,
        ).to(device)

        # 保存配置文件
        config_file = os.path.join(snapshot_path, 'config.txt')
        with open(config_file, 'w', encoding='utf-8') as f:
            for key, value in args.__dict__.items():
                f.write(f'{key}: {value}\n')

        # 启动当前参数组的训练（会自动从 latest_model.pth 恢复）
        train_model(args, net, snapshot_path)
        print(f"第 {idx} 组实验处理完毕！\n")

    # print("所有 7 组消融实验已全部顺利执行完毕！")