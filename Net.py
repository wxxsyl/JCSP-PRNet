import torch
import math
import torch.nn as nn
from einops import repeat
from entmax import Sparsemax
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class SharedExpert(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class SpatialAttentionExpert(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.feat_extract = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
            nn.GELU()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.out_proj = nn.Conv2d(ch, ch, kernel_size=1, bias=False)

    def forward(self, x):
        feat = self.feat_extract(x)
        max_feat = torch.max(feat, dim=1, keepdim=True).values
        avg_feat = torch.mean(feat, dim=1, keepdim=True)
        spatial_attn = self.spatial_gate(torch.cat([max_feat, avg_feat], dim=1))
        return self.out_proj(feat * spatial_attn)


class PositionRouter(nn.Module):
    def __init__(self, ch, num_routed_experts=3, router_ch=None):
        super().__init__()
        mid = router_ch if router_ch is not None else max(8, ch // 8)
        if min(ch, mid, num_routed_experts) < 1:
            raise ValueError('Channels and expert count must be positive')
        self.num_routed_experts = num_routed_experts
        self.proj = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, num_routed_experts, 1),
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid, num_routed_experts, 1, bias=False),
        )
        self.sparsemax = Sparsemax(dim=1)

    def forward(self, x):
        z = self.proj(x)

        logits_local = self.local(z)
        logits_global = self.global_branch(z)

        logits = logits_local * logits_global

        return self.sparsemax(logits)




class PRMoEBlock(nn.Module):
    def __init__(self, ch, num_routed_experts=3, router_ch=None):
        super().__init__()
        if num_routed_experts < 1:
            raise ValueError('num_routed_experts must be >= 1')
        self.num_routed_experts = num_routed_experts
        self.pre_conv = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GELU()
        )
        self.shared_expert = SharedExpert(ch)
        self.routed_experts = nn.ModuleList([
            SpatialAttentionExpert(ch) for _ in range(num_routed_experts)
        ])
        self.router = PositionRouter(ch, num_routed_experts, router_ch=router_ch)

    def forward(self, x):
        x_pre = self.pre_conv(x)
        routing_weights = self.router(x_pre)
        routed_detail = torch.zeros_like(x_pre)
        for i, expert in enumerate(self.routed_experts):
            routed_detail = routed_detail + expert(x_pre) * routing_weights[:, i:i + 1]
        shared_detail = self.shared_expert(x_pre)
        return shared_detail + routed_detail


class PRMoE_Encoder(nn.Module):
    def __init__(self, in_ch=16, stage_channels=None, num_experts=3):
        super().__init__()
        if stage_channels is None:
            stage_channels = [32, 64, 96, 128]
        self.stages = len(stage_channels)
        self.down_convs = nn.ModuleList()
        self.moe_blocks = nn.ModuleList()
        prev_ch = in_ch
        for i, out_ch in enumerate(stage_channels):
            stride = 1 if i == 0 else 2
            self.down_convs.append(nn.Sequential(
                nn.Conv2d(prev_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ))
            self.moe_blocks.append(PRMoEBlock(out_ch, num_routed_experts=num_experts))
            prev_ch = out_ch

    def forward(self, x):
        feats = []
        for down_conv, moe in zip(self.down_convs, self.moe_blocks):
            x = moe(down_conv(x))
            feats.append(x)
        return feats



class JCSSM(nn.Module):
    def __init__(self, d_inner, dt_rank, k, d_state=16, dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.K = k

        # 1. x projection
        x_proj = [nn.Linear(d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs) for _ in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in x_proj], dim=0))
        del x_proj

        # 2. dt projection
        dt_projs = [self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs

        # 3. State transition parameters A and D
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K, merge=True, device=device)
        self.Ds = self.D_init(d_inner, copies=self.K, merge=True, device=device)

        # 4. Selective scan
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A_log = torch.log(A).unsqueeze(0).repeat(d_inner, 1).contiguous()
        if copies > 0:
            A_log = A_log.unsqueeze(0).repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, dtype=torch.float32, device=device)
        if copies > 0:
            D = D.unsqueeze(0).repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor):
        B, K, D, L = x.shape
        x_dtype, y_dtype = x.dtype, y.dtype

        # 1. Generate shared selective parameters from cond
        cond_dbl = torch.einsum("b k d l, k c d -> b k c l", cond, self.x_proj_weight)
        dts, Bs, Cs = torch.split(cond_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # 2. Prepare sequences
        xs = x.reshape(B, self.K * self.d_inner, L).contiguous().float()
        ys = y.reshape(B, self.K * self.d_inner, L).contiguous().float()
        dts = dts.reshape(B, self.K * self.d_inner, L).contiguous().float()
        Bs = Bs.contiguous().float()
        Cs = Cs.contiguous().float()

        # 3. Shared A, D and dt bias
        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float().reshape(-1)

        # 4. Scan IR and VI branches
        out_x = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False)
        out_y = self.selective_scan(ys, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False)

        # 5. Restore dimensions and dtypes
        out_x = out_x.reshape(B, self.K, self.d_inner, L).to(x_dtype)
        out_y = out_y.reshape(B, self.K, self.d_inner, L).to(y_dtype)

        return out_x, out_y




class JCSP(nn.Module):
    def __init__(self, d_model, d_conv=3):
        super().__init__()
        self.d_model = d_model
        self.k = 2

        # 1. Modality-specific normalization and projection
        self.norm_1 = nn.GroupNorm(1, d_model)
        self.norm_2 = nn.GroupNorm(1, d_model)
        self.proj1 = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(d_model, d_model, kernel_size=1, bias=False)

        # 2. Modality-specific local feature extraction
        self.conv2d1 = nn.Conv2d(d_model, d_model, kernel_size=d_conv, padding=d_conv // 2, groups=d_model, bias=True)
        self.conv2d2 = nn.Conv2d(d_model, d_model, kernel_size=d_conv, padding=d_conv // 2, groups=d_model, bias=True)
        self.act = nn.SiLU()

        # 3. Joint condition generation (输出通道与 d_model 一致，用于生成 0~1 的动态融合比率)
        self.cond_proj = nn.Conv2d(d_model * 2, d_model, kernel_size=1, bias=False)

        # 4. Joint-conditioned SSM
        d_state = 16
        dt_rank = max(1, d_model // 16)
        self.ssm = JCSSM(d_inner=d_model, dt_rank=dt_rank, k=self.k, d_state=d_state)


        self.out_proj = nn.Conv2d(d_model*2, d_model, kernel_size=1, bias=False)

    def spatial_to_seq_2dir(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2)
        fwd = x_seq
        bwd = torch.flip(x_seq, dims=[-1])
        return torch.stack([fwd, bwd], dim=1).contiguous()

    def seq_to_spatial_2dir(self, out_seq, H, W):
        B, K, C, L = out_seq.shape
        y_fwd = out_seq[:, 0]
        y_bwd = torch.flip(out_seq[:, 1], dims=[-1])
        y = y_fwd + y_bwd
        return y.reshape(B, C, H, W).contiguous()

    def forward(self, x, y):
        if x.shape != y.shape:
            raise ValueError(
                f"IR and VI features must have the same shape, "
                f"got {tuple(x.shape)} and {tuple(y.shape)}"
            )

        B, C, H, W = x.shape
        res_x, res_y = x, y

        # 1. Modality-specific feature extraction
        x_n = self.proj1(self.norm_1(x))
        y_n = self.proj2(self.norm_2(y))
        x_main = self.act(self.conv2d1(x_n))
        y_main = self.act(self.conv2d2(y_n))

        cond_feat =  self.cond_proj(torch.cat([x_main, y_main], dim=1))

        # 3. Bidirectional sequence construction
        x_seq = self.spatial_to_seq_2dir(x_main)
        y_seq = self.spatial_to_seq_2dir(y_main)
        cond_seq = self.spatial_to_seq_2dir(cond_feat)

        out_x_seq, out_y_seq = self.ssm(x_seq, y_seq, cond_seq)

        out_x = self.seq_to_spatial_2dir(out_x_seq, H, W) + res_x 
        out_y = self.seq_to_spatial_2dir(out_y_seq, H, W) + res_y

        out = self.out_proj(torch.cat((out_x, out_y), dim=1))

        return out




class UpsampleFuseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GELU()
        )
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.GELU()
        )

        self.residual = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )

    def forward(self, same_level_feats, up_feat):
        upsampled = self.upsample(up_feat)

        fused_sum = upsampled
        for feat in same_level_feats:
            fused_sum = fused_sum + feat

        fused = self.fuse(fused_sum)

        return self.residual(fused) + fused


class NestDecoder(nn.Module):
    def __init__(self, channels=[32, 64, 96, 128], out_ch=1):
        super().__init__()
        self.stages = len(channels)
        self.channels = channels

        self.decoder_nodes = nn.ModuleDict()
        for j in range(1, self.stages):
            for i in range(self.stages - j):
                name = f"{i}_{j}"
                self.decoder_nodes[name] = UpsampleFuseBlock(
                    in_ch=self.channels[i + 1],
                    out_ch=self.channels[i]
                )

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.channels[0], self.channels[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.channels[0], out_ch, kernel_size=3, padding=1)
        )

    def forward(self, feats):
        dense_feats = [[None for _ in range(self.stages)] for _ in range(self.stages)]

        for i in range(self.stages):
            dense_feats[i][0] = feats[i]

        for j in range(1, self.stages):
            for i in range(self.stages - j):
                name = f"{i}_{j}"
                same_level_feats = [dense_feats[i][k] for k in range(j)]
                up_feat = dense_feats[i + 1][j - 1]
                dense_feats[i][j] = self.decoder_nodes[name](same_level_feats, up_feat)

        return self.final_conv(dense_feats[0][self.stages - 1])


class FusionNet(nn.Module):
    def __init__(self, in_ch=1, stem_ch=32, stage_channels=None,
                 num_experts=3):
        super().__init__()
        if stage_channels is None:
            stage_channels = [32, 64, 96, 128]
        stage_channels = list(stage_channels)
        self.stages = len(stage_channels)

        self.stem_ir = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1, bias=False), nn.GELU()
        )
        self.stem_vi = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, padding=1, bias=False), nn.GELU(),
            nn.Conv2d(stem_ch, stem_ch, 3, padding=1, bias=False), nn.GELU()
        )
        encoder_kwargs = dict(
            in_ch=stem_ch, stage_channels=stage_channels, num_experts=num_experts
        )
        self.encoder_ir = PRMoE_Encoder(**encoder_kwargs)
        self.encoder_vi = PRMoE_Encoder(**encoder_kwargs)
        
        self.cross_fusion = nn.ModuleList([
            JCSP(d_model=stage_channels[i])
            for i in range(self.stages)
        ])
        self.decoder = NestDecoder(channels=stage_channels, out_ch=1)

    def forward(self, ir, vi):
        ir_feats = self.encoder_ir(self.stem_ir(ir))
        vi_feats = self.encoder_vi(self.stem_vi(vi))
        fused_feats = [fusion(x, y) for fusion, x, y
                       in zip(self.cross_fusion, ir_feats, vi_feats)]
        return self.decoder(fused_feats)




if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("CUDA unavailable: skipping the native selective_scan smoke test.")
    else:
        device = torch.device("cuda")
        model = FusionNet().to(device).eval()
        ir = torch.randn(1, 1, 128, 128, device=device)
        vi = torch.randn(1, 1, 128, 128, device=device)
        with torch.no_grad():
            fused = model(ir, vi)
        print(f"IR shape: {tuple(ir.shape)}")
        print(f"VI shape: {tuple(vi.shape)}")
        print(f"Fused shape: {tuple(fused.shape)}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.4f} M")