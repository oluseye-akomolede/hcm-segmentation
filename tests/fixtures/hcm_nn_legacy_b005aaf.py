import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import math

ISC_OUTPUT_CHANNEL_NUM = 32
ISC_INNER_CHANNEL_NUMBER = 64
ISC_INPUT_CHANNEL_NUMBER = 256
RESNET_OUTPUT_CHANNEL_NUMBER = 2048
RRD_OUTPUT_CHANNEL_NUMBER = 32
ASPP_RATES = [6,12,18]

def _check(name, t):
    finite = torch.isfinite(t)
    if not finite.all():
        bad = t[~finite].flatten()[:5]
        tf = t[finite]
        mn = tf.min().item() if tf.numel() else float("nan")
        mx = tf.max().item() if tf.numel() else float("nan")
        print(f"[NaN/Inf] {name}: shape={tuple(t.shape)} finite_min={mn} finite_max={mx} bad_sample={bad}")
        raise RuntimeError(f"Non-finite detected in {name}")


# The largest magnitude that may be handed to code running under autocast/fp16.
# fp16 tops out at 65504, and the cast saturates silently: 70000.0 in fp32 is
# finite, and `.to(float16)` turns it into inf with no warning at all. Leave
# headroom below the ceiling so the boundary trips before a value is one
# rounding step away from saturating.
FP16_MAX_SAFE = 6.0e4


def _guard_fp16_boundary(name, t):
    """Refuse to hand a tensor of unsafe magnitude to fp16/autocast code.

    Raises rather than clamps: clamping would hide real activation growth from
    the owner, which is the thing this whole path exists to make visible. A
    value that trips this is a genuine numerical problem worth seeing, not
    something to quietly flatten.
    """
    mx = t.detach().abs().amax().item()
    if not math.isfinite(mx) or mx > FP16_MAX_SAFE:
        raise RuntimeError(
            f"{name} max|x|={mx:.6g} exceeds the safe fp16 boundary "
            f"{FP16_MAX_SAFE:.6g} (fp16 goes non-finite above 65504); "
            f"casting here would overflow to inf"
        )
    return t


# Stages whose activation magnitude is watched. These values are plain
# magnitudes, not percentages -- they must stay out of any 0-100 range guard.
#
# "isc3" is isc_output[3] specifically: it is monitored on its own because it is
# the only ISC output that reaches the decoder (rrd_list[3] takes it directly),
# so it is the stage whose growth the fp16 boundary actually depends on. The
# aggregate "isc" is the max over all four ISC outputs, which can hide stage 3
# sitting at the ceiling while the other three stay small.
ACTIVATION_STAGES = ("isc", "csc", "isc3")


def _record_absmax(module, stage, t):
    """Fold max|t| into the module's running max for ``stage``.

    Detached and host-side: this is a diagnostic, so it must not carry a
    gradient or hold the graph alive. The caller (the trainer) reads and resets
    it at each logging point, so the running max catches a single spike between
    logging points rather than only sampling the batch that happens to land on
    one.
    """
    try:
        mx = t.detach().abs().amax().item()
    except (RuntimeError, AttributeError):
        # Never let a diagnostic break training.
        return
    if not isinstance(getattr(module, "activation_absmax", None), dict):
        module.activation_absmax = {s: 0.0 for s in ACTIVATION_STAGES}
    if mx > module.activation_absmax.get(stage, 0.0):
        module.activation_absmax[stage] = float(mx)


def group_norm(channels, num_groups=32):
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels)

def modify_resnet50_stride(model,stride_layer=2):
    # Modify the first convolutional layer
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
    
    # Modify the downsampling layers in the residual blocks
    strides = [1, 1, 1, 1]
    strides[stride_layer-1] = 1
    
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        for block in layer:
            block.conv1.stride = (1, 1)
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        if i < 4:
            layer[0].conv1.stride = (strides[i], strides[i])
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (strides[i], strides[i])
    return model

def modify_resnet50_stride_and_channels(model,stride_layer=2):
    # Modify the first convolutional layer
    model.conv1 = torch.nn.Conv2d(2048, 64, kernel_size=7, stride=1, padding=3, bias=False)
    
    # Modify the downsampling layers in the residual blocks
    strides = [1, 1, 1, 1]
    strides[stride_layer-1] = 1
    
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
        for block in layer:
            block.conv1.stride = (1, 1)
            block.conv2.stride = (1, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (1, 1)
        if i < 4:
            layer[0].conv1.stride = (strides[i], strides[i])
            if layer[0].downsample is not None:
                layer[0].downsample[0].stride = (strides[i], strides[i])
    return model

class hcm_isc(nn.Module):
    def __init__(self):
        super(hcm_isc,self).__init__()
        self.conv1_1_1x1 = nn.Conv2d(RESNET_OUTPUT_CHANNEL_NUMBER,ISC_OUTPUT_CHANNEL_NUM,kernel_size=1,padding=0)
        self.conv1_2_1x1 = nn.Conv2d(RESNET_OUTPUT_CHANNEL_NUMBER,ISC_INPUT_CHANNEL_NUMBER,kernel_size=1,padding=0)
        self.conv1_3_1x1 = nn.Conv2d(RESNET_OUTPUT_CHANNEL_NUMBER,ISC_INPUT_CHANNEL_NUMBER,kernel_size=1,padding=0)
        self.conv2_1_3x3 = nn.Conv2d(ISC_INPUT_CHANNEL_NUMBER,ISC_INNER_CHANNEL_NUMBER,kernel_size=3,padding=1)
        self.conv2_2_5x5 = nn.Conv2d(ISC_INPUT_CHANNEL_NUMBER,ISC_INNER_CHANNEL_NUMBER,kernel_size=5, padding=2)
        self.conv3_1_3x3 = nn.Conv2d(ISC_INNER_CHANNEL_NUMBER,ISC_INNER_CHANNEL_NUMBER,kernel_size=3,padding=1)
        self.conv3_2_5x5 = nn.Conv2d(ISC_INNER_CHANNEL_NUMBER,ISC_INNER_CHANNEL_NUMBER,kernel_size=5, padding=2)
        self.conv4_1_3x3 = nn.Conv2d(ISC_INNER_CHANNEL_NUMBER,ISC_OUTPUT_CHANNEL_NUM,kernel_size=3,padding=1)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4  # 4 bytes per parameter (float32)
        return total_size / (1024 * 1024)  # Convert to MB

    def forward(self, x):
        _check("isc_in", x)
        x = x.float()

        out1_1 = self.conv1_1_1x1(x)
        out1_2 = self.conv1_2_1x1(x)
        out1_3 = self.conv1_3_1x1(x)

        out_2_1 = self.conv2_1_3x3(out1_2)
        _check("out_2_1", out_2_1)
        out_2_2 = self.conv2_2_5x5(out1_3)

        int_out_2 = out_2_1 + out_2_2
        _check("int_out_2", int_out_2)

        out3_1 = self.conv3_1_3x3(int_out_2)
        _check("out3_1", out3_1)
        out3_2 = self.conv3_2_5x5(int_out_2)
        _check("out3_2", out3_2)

        int_out_3 = out3_1 * out3_2
        int_out_3 = torch.nan_to_num(int_out_3, nan=0.0, posinf=1e4, neginf=-1e4)
        int_out_3 = int_out_3.clamp(-1e4, 1e4)

        int_out_3 = F.group_norm(int_out_3, num_groups=8, eps=1e-5)
        _check("int_out_3", int_out_3)

        out4_1 = self.conv4_1_3x3(int_out_3)
        _check("out4_1", out4_1)

        output = out1_1 + out4_1
        _check("output", output)

        # Return fp32, NOT the input dtype. `output` is unbounded after the
        # group_norm above, and casting it back to fp16 saturates silently once
        # |output| > 65504: a real run died here with a -inf in isc_out[0] that
        # `_check` had just passed, because the check ran on the fp32 value and
        # the very next statement is what produced the inf. The caller keeps
        # this in fp32 and guards the boundary where fp16 resumes.
        return output

class JointAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(JointAttention,self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
        B, C, H, W = x.size()
        if H > 16 or W > 16:
            x = F.adaptive_avg_pool2d(x, (16, 16))
        head_dim = C // self.num_heads
        assert C % self.num_heads == 0, f"C={C} not divisible by heads={self.num_heads}"

        q = self.query(x).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)  # (B, h, HW, d)
        k = self.key(x).view(B, self.num_heads, head_dim, H * W).transpose(2, 3)    # (B, h, HW, d)
        v = self.value(x).view(B, self.num_heads, head_dim, H * W)                  # (B, h, d, HW)
        _check("JA_x", x)
        _check("JA_q", q)
        _check("JA_k", k)
        _check("JA_v", v)

        # ---- critical stability block: do scores + softmax in fp32 ----
        qf = q.float()
        kf = k.float()
        vf = v.float()

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(qf, kf.transpose(2, 3)) * scale          # (B, h, HW, HW)
        
        # stability trick used everywhere
        scores = scores - scores.max(dim=-1, keepdim=True).values

        attn = F.softmax(scores, dim=-1)                               # fp32 softmax
        out = torch.matmul(attn, vf.transpose(2, 3))                   # (B, h, HW, d)
        out = out.to(dtype=q.dtype)                                    # cast back (fp16/bf16)

        if not torch.isfinite(scores).all():
            print("scores min/max:", scores.min().item(), scores.max().item())
            raise RuntimeError("Non-finite scores")

        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        return self.out_proj(out)

class PONO(torch.nn.Module):
    def __init__(self, num_channels):
        super(PONO,self).__init__()
        self.num_channels = num_channels
        self.weight = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    
    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
         # x: (B, C, H, W)
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True, unbiased=False)
        std = std.clamp_min(1e-5)
        normalized = (x - mean) / std
        return self.weight * normalized + self.bias

class hcm_csc(nn.Module):
    def __init__(self):
        super(hcm_csc,self).__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=ISC_OUTPUT_CHANNEL_NUM)
        self.pre_attn_norm = nn.GroupNorm(num_groups=8, num_channels=ISC_OUTPUT_CHANNEL_NUM)
        self.JA = JointAttention(in_channels=ISC_OUTPUT_CHANNEL_NUM)
        self.PN = PONO(ISC_OUTPUT_CHANNEL_NUM)
    
    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, f_s, f_splus1):
        _check("f_s", f_s)
        _check("f_splus1", f_splus1)
        f_s = self.norm(f_s)
        _check("normed_f_s", f_s)
        f_splus1 = self.norm(f_splus1)
        _check("normed_f_splus1", f_splus1)
        upsampled = F.interpolate(
            f_splus1,
            size=f_s.shape[-2:],     # ✅ match spatial dims exactly
            mode="bilinear",
            align_corners=False
        )
        _check("upsampled", upsampled)
        added_input = torch.add(f_s, upsampled)
        _check("added_input", added_input)
        # HARD CAP before attention — this is mandatory
        added_input_ds = F.adaptive_avg_pool2d(added_input, (16, 16))

        added_input_ds = self.pre_attn_norm(added_input_ds)

        _check("added_input_ds", added_input_ds)
        joint = self.JA(added_input_ds)
        _check("JA_out", joint)
        joint = F.interpolate(joint, size=added_input.shape[-2:], mode="bilinear", align_corners=False)
        normalized = self.PN(joint)
        _check("PONO_out", normalized)
        return normalized

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        # Channel Attention
        attn = self.avg_pool(out)
        attn = attn.view(attn.size(0), -1)
        attn = self.fc1(attn)
        attn = self.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        attn = attn.view(attn.size(0), attn.size(1), 1, 1)
        
        out = out * attn
        out = out + residual
        return out
    
class hcm_rrd(nn.Module):
    def __init__(self):
        super().__init__()
        self.RCAB = ResidualChannelAttentionBlock(ISC_OUTPUT_CHANNEL_NUM)
        self.conv2d = nn.Conv2d(ISC_OUTPUT_CHANNEL_NUM,RRD_OUTPUT_CHANNEL_NUMBER,kernel_size=3,padding=1,stride=(1,2))

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self,f_s:torch.Tensor,p_splus1:torch.Tensor):
        repeated_p_s = F.interpolate(
            p_splus1,
            size=f_s.shape[-2:],     # ✅ match f_s exactly
            mode="bilinear",
            align_corners=False
        )
        sig_p_s = F.sigmoid(repeated_p_s)
        fp_mult_normal = torch.mul(f_s,sig_p_s)
        fp_mult_normal = F.group_norm(fp_mult_normal, num_groups=8, eps=1e-5)

        sig_p_s_reversed = torch.sub(1,sig_p_s)
        fp_mult_reversed = torch.mul(f_s,sig_p_s_reversed)
        fp_mult_reversed = F.group_norm(fp_mult_reversed, num_groups=8, eps=1e-5)

        cat_tensor = torch.cat((fp_mult_normal,fp_mult_reversed),dim=3)
        rcab_output = self.RCAB(cat_tensor)
        conv_output = self.conv2d(rcab_output)
        return conv_output
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1x1 = group_norm(out_channels)

        self.conv_rates = nn.ModuleList()
        self.bn_rates = nn.ModuleList()
        for rate in rates:
            self.conv_rates.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate))
            self.bn_rates.append(group_norm(out_channels))

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            group_norm(out_channels),
            nn.ReLU()
        )

        self.conv_concat = nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, stride=(2, 2))
        self.bn_concat = group_norm(out_channels)

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, x):
        res = [F.relu(self.bn1x1(self.conv1x1(x)))]
        for conv, bn in zip(self.conv_rates, self.bn_rates):
            res.append(F.relu(bn(conv(x))))
        pool = F.interpolate(self.global_avg_pool(x), size=x.shape[2:], mode='bilinear', align_corners=True)
        res.append(pool)
        x = torch.cat(res, dim=1)
        x = F.relu(self.bn_concat(self.conv_concat(x)))
        return x

class DifferentiableROIModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()  # [B, 1, H, W] — attention map
        )

    def get_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_size = total_params * 4
        return total_size / (1024 * 1024)

    def forward(self, features):
        attn_map = self.attn(features)
        return attn_map

def _check_tensor(name, t):
    if not torch.is_tensor(t):
        print(f"{name}: not a tensor")
        return
    finite = torch.isfinite(t)
    if not finite.all():
        bad = t.flatten()[~finite.flatten()][:5]
        print(f"[NaN/Inf] {name}: shape={tuple(t.shape)} "
              f"min={t.min().item()} max={t.max().item()} bad_sample={bad}")
        raise RuntimeError(f"Non-finite detected in {name}")
    # optional: also print magnitude diagnostics
    t_absmax = t.abs().max().item()
    print(f"[OK] {name}: shape={tuple(t.shape)} min={t.min().item():.4g} "
          f"max={t.max().item():.4g} absmax={t_absmax:.4g} dtype={t.dtype}")

def _gn32(t, groups=32, eps=1e-5):
    # groups must divide channels
    c = t.shape[1]
    g = min(groups, c)
    while c % g != 0 and g > 1:
        g -= 1
    return F.group_norm(t, num_groups=g, eps=eps)

class ResNet50Backbone(nn.Module):
    def __init__(self, out_stride=16, pretrained=True):
        super().__init__()
        assert out_stride in (8, 16, 32)

        # replace_stride_with_dilation corresponds to layer2, layer3, layer4
        # default OS=32 (no dilation)
        if out_stride == 32:
            dil = (False, False, False)
        elif out_stride == 16:
            dil = (False, False, True)   # keep layer4 resolution, use dilation there
        else:  # out_stride == 8
            dil = (False, True, True)    # keep layer3+layer4 resolution

        m = models.resnet50(
            weights=ResNet50_Weights.DEFAULT if pretrained else None,
            replace_stride_with_dilation=dil
        )

        # "stem"
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool

        # stages
        self.layer1 = m.layer1  # C2
        self.layer2 = m.layer2  # C3
        self.layer3 = m.layer3  # C4
        self.layer4 = m.layer4  # C5

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c2 = self.layer1(x)   # ~1/4
        c3 = self.layer2(c2)  # ~1/8
        c4 = self.layer3(c3)  # ~1/16 (or ~1/8 if OS=8)
        c5 = self.layer4(c4)  # ~1/32 (or ~1/16/~1/8 if dilated)
        return c2, c3, c4, c5

class hcm_nn(nn.Module):
    def __init__(self, enable_finetuning: bool = False, num_segmentation_classes: int = None, out_stride=16):
        super().__init__()
        self.backbone = ResNet50Backbone(out_stride=out_stride, pretrained=True)
        self.backbone = self.backbone.float()

        # adapt each stage to 2048 so your existing ISC blocks can stay unchanged
        self.adapt_c2 = nn.Conv2d(256, 2048, kernel_size=1)
        self.adapt_c3 = nn.Conv2d(512, 2048, kernel_size=1)
        self.adapt_c4 = nn.Conv2d(1024, 2048, kernel_size=1)
        self.adapt_c5 = nn.Identity()  # already 2048

        self.ASPP = ASPP(2048, ISC_OUTPUT_CHANNEL_NUM, ASPP_RATES)

        self.isc_list = nn.ModuleList([hcm_isc() for _ in range(4)])
        self.csc_list = nn.ModuleList([hcm_csc() for _ in range(3)])
        self.rrd_list = nn.ModuleList([hcm_rrd() for _ in range(4)])
        self.differentiable_roi_module = DifferentiableROIModule(ISC_OUTPUT_CHANNEL_NUM)

        self.roi_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduced spatial size
            nn.Flatten(),
            nn.Linear(ISC_OUTPUT_CHANNEL_NUM, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1000)
        )

        #nn.init.kaiming_normal_(self.roi_classifier[2].weight, mode='fan_out', nonlinearity='relu')
        #nn.init.zeros_(self.roi_classifier[-1].bias)

        if num_segmentation_classes is not None:
            self.segmentation_head = nn.Conv2d(ISC_OUTPUT_CHANNEL_NUM, num_segmentation_classes, kernel_size=1)
        else:
            self.segmentation_head = None

        self.enable_finetuning = enable_finetuning
        # Recorded so the uploader's model_info.pkl carries real values; without
        # them it pickled None and the loader could not rebuild the model.
        self.out_stride = out_stride
        self.num_segmentation_classes = num_segmentation_classes

        # Running max|activation| per stage since the last reset; the trainer
        # reads and clears this at each logging point. Plain magnitudes, not
        # percentages -- they must not meet the 0-100 metric guard.
        self.activation_absmax = {s: 0.0 for s in ACTIVATION_STAGES}

    def reset_activation_absmax(self):
        """Clear the running activation maxima (called after each log)."""
        for stage in self.activation_absmax:
            self.activation_absmax[stage] = 0.0

    def resnet_plus_norm(self, mod, inp):
        out = mod(inp)
        out = _gn32(out.float()).to(out.dtype)
        return out

    def forward_internal(self, x):
        _check("model_in_x", x)
        d_type = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            c2, c3, c4, c5 = self.backbone(x32)
            _check("c2", c2); _check("c3", c3); _check("c4", c4); _check("c5", c5)
        
        c2 = c2.to(d_type)
        c3 = c3.to(d_type)
        c4 = c4.to(d_type)
        c5 = c5.to(d_type)

        # Make a consistent pyramid, all as 2048-ch maps
        r2 = self.adapt_c2(c2); _check("r2", r2)
        r3 = self.adapt_c3(c3); _check("r3", r3)
        r4 = self.adapt_c4(c4); _check("r4", r4)
        r5 = self.adapt_c5(c5); _check("r5", r5)

        # Optional: pick a "working resolution" (e.g., r4 spatial size)
        target_hw = r2.shape[-2:]
        r2 = F.interpolate(r2, size=target_hw, mode="bilinear", align_corners=False)
        r3 = F.interpolate(r3, size=target_hw, mode="bilinear", align_corners=False)
        r5 = F.interpolate(r5, size=target_hw, mode="bilinear", align_corners=False)

        # Now you have 4 depth-aligned feature maps that "work at all depths"
        res_output = [r2, r3, r4, r5]

        isc_output = []
        # The whole ISC->CSC chain runs outside autocast, in fp32. The CSC stage
        # must be INSIDE this block, not merely handed fp32 tensors: autocast
        # casts a conv's *inputs* down to fp16 even when the caller passes fp32,
        # so a CSC loop sitting just outside here would silently lose the range
        # one step later and overflow exactly as the ISC cast used to.
        with torch.cuda.amp.autocast(enabled=False):
            for i in range(4):
                t = self.isc_list[i](res_output[i].float())
                _check(f"isc_out[{i}]", t)
                _record_absmax(self, "isc", t)
                if i == 3:
                    # Tracked separately: this is the only ISC output that
                    # reaches the decoder, so it is the one the rrd boundary
                    # below has to survive.
                    _record_absmax(self, "isc3", t)
                isc_output.append(t)

            csc_output = [None, None, None]
            for i in range(2, -1, -1):
                if i != 2:
                    csc_output[i] = self.csc_list[i](isc_output[i], csc_output[i + 1])
                else:
                    csc_output[i] = self.csc_list[i](isc_output[i], isc_output[i + 1])
                _record_absmax(self, "csc", csc_output[i])

        aspp_output = self.ASPP(res_output[3])

        del res_output
        # The single guarded boundary. Everything above runs fp32; everything
        # from here on runs under the caller's autocast, so these fp32 values
        # get cast down to fp16 for the decoder convs. That cast is safe only
        # while the magnitudes are, so refuse loudly rather than let it saturate
        # into the inf that killed the run.
        #
        # rrd_list[3] is the one rung that consumes a RAW ISC output
        # (isc_output[3]); every other rung consumes a CSC output, which is
        # already small. The ISC outputs are the largest tensors in the model
        # (smoke runs saw ~41k against an fp16 ceiling of 65504), so this rung
        # must run outside autocast for the same reason the CSC loop above does:
        # autocast would cast the conv's *inputs* down to fp16 and lose the
        # range one step before the guard below could see it.
        rrd_output = [None, None, None, None]
        for i in range(3, -1, -1):
            if i != 3:
                rrd_output[i] = self.rrd_list[i](csc_output[i], rrd_output[i + 1])
            else:
                with torch.cuda.amp.autocast(enabled=False):
                    rrd_output[i] = self.rrd_list[i](
                        isc_output[i].float(), aspp_output.float()
                    )
            _guard_fp16_boundary(f"rrd_out[{i}]", rrd_output[i])

        del isc_output, csc_output, aspp_output

        output = rrd_output[0]
        #output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)

        if self.enable_finetuning:
            return self.segmentation_head(output) if self.segmentation_head else output

        attn_map = self.differentiable_roi_module(output)
        weighted_features = output * attn_map
        pooled = self.roi_classifier[0](weighted_features)
        flattened = self.roi_classifier[1](pooled)
        logits = self.roi_classifier[2:](flattened)

        return logits

    def forward(self, x):
        return self.forward_internal(x)
    
    def get_backbone_names(self):
        return [name for name, _ in self.decoder_resnet50s.named_parameters()]
    
    @staticmethod
    def convert_batchnorm_to_groupnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, group_norm(child.num_features))
            else:
                hcm_nn.convert_batchnorm_to_groupnorm(child)


def checkpoint_residual_block(block):
    class CheckpointWrapper(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x):
            return self.block(x)

    return CheckpointWrapper(block)




        