import torch
import torch.nn as nn
import torch.nn.functional as F

def center_crop(t, target_h, target_w):
    # t: tensor with shape [B, C, H, W]
    _, _, H, W = t.shape
    start_h = (H - target_h) // 2
    start_w = (W - target_w) // 2
    return t[:, :, start_h:start_h+target_h, start_w:start_w+target_w]

def match_size(a, b):
    # Crop both a and b to the minimum H and W.
    _, _, H_a, W_a = a.shape
    _, _, H_b, W_b = b.shape
    target_H = min(H_a, H_b)
    target_W = min(W_a, W_b)
    a_cropped = a[:, :, :target_H, :target_W]
    b_cropped = b[:, :, :target_H, :target_W]
    return a_cropped, b_cropped

class FFCSE_block(nn.Module):
    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r, kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is a tuple (x_local, x_global)
        x = x if isinstance(x, tuple) else (x, 0)
        id_l, id_g = x
        x_cat = id_l if isinstance(id_g, int) else torch.cat([id_l, id_g], dim=1)
        x_pool = self.avgpool(x_cat)
        x_conv = self.relu1(self.conv1(x_pool))
        x_l = 0 if self.conv_a2l is None else id_l * self.sigmoid(self.conv_a2l(x_conv))
        x_g = 0 if self.conv_a2g is None else id_g * self.sigmoid(self.conv_a2g(x_conv))
        return x_l, x_g

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        # Process entire channel group (no grouping)
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        B, C, H, W = x.size()
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        B, C, H_pad, W_pad = x.size()
        ffted = torch.fft.rfft2(x, norm="ortho")
        ffted = torch.view_as_real(ffted)  # [B, C, H_pad, W_pad//2+1, 2]
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, 2, H_pad, W_pad//2+1]
        ffted = ffted.view(B, C * 2, H_pad, W_pad//2+1)
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))
        out_channels = ffted.size(1) // 2
        ffted = ffted.view(B, out_channels, 2, H_pad, W_pad//2+1)
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()  # [B, out_channels, H_pad, W_pad//2+1, 2]
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(H_pad, W_pad), norm="ortho")
        output = output[:, :, :H, :W]
        return output



# class DualFourierAttention(nn.Module):
#     def __init__(self, in_channels_local, in_channels_global):
#         super(DualFourierAttention, self).__init__()
#         self.conv_local = nn.Conv2d(in_channels_local, in_channels_local, kernel_size=1)
#         self.conv_global = nn.Conv2d(in_channels_global, in_channels_global, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, Y_local, Y_global):
#         # Generate attention maps
#         attn_global = self.sigmoid(self.conv_global(Y_global))
#         attn_local = self.sigmoid(self.conv_local(Y_local))

#         # Ensure attn_global matches Y_local spatially
#         if attn_global.shape[-2:] != Y_local.shape[-2:]:
#             attn_global = F.interpolate(attn_global, size=Y_local.shape[-2:], mode='bilinear', align_corners=False)

#         # Ensure attn_local matches Y_global spatially
#         if attn_local.shape[-2:] != Y_global.shape[-2:]:
#             attn_local = F.interpolate(attn_local, size=Y_global.shape[-2:], mode='bilinear', align_corners=False)

#         # Modulate features
#         Y_local_modulated = Y_local * attn_global
#         Y_global_modulated = Y_global * attn_local

#         return Y_local_modulated, Y_global_modulated


class DualFourierAttention(nn.Module):
    def __init__(self, in_channels_local, in_channels_global):
        super(DualFourierAttention, self).__init__()
        self.norm_l = nn.BatchNorm2d(in_channels_local)
        self.norm_g = nn.BatchNorm2d(in_channels_global)

        self.attn_local = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_local, in_channels_local // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_local // 2, in_channels_local, 1),
            nn.Sigmoid()
        )
        self.attn_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_global, in_channels_global // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels_global // 2, in_channels_global, 1),
            nn.Sigmoid()
        )

        # Fusion weights (learned importance of each path)
        self.fusion_weight = nn.Linear(in_channels_local + in_channels_global, 2)

    def forward(self, Y_l, Y_g):
        Y_l = self.norm_l(Y_l)
        Y_g = self.norm_g(Y_g)

        # Attention weights
        A_g = self.attn_global(Y_g)
        A_l = self.attn_local(Y_l)

        # Resize attention if needed
        if A_g.shape[-2:] != Y_l.shape[-2:]:
            A_g = F.interpolate(A_g, size=Y_l.shape[-2:], mode='bilinear', align_corners=False)
        if A_l.shape[-2:] != Y_g.shape[-2:]:
            A_l = F.interpolate(A_l, size=Y_g.shape[-2:], mode='bilinear', align_corners=False)

        # Apply attention modulation
        Y_l_mod = Y_l * A_g + Y_l  # Residual path
        Y_g_mod = Y_g * A_l + Y_g  # Residual path

        # # Global average pool + fusion
        # z_l = F.adaptive_avg_pool2d(Y_l_mod, 1).view(Y_l_mod.size(0), -1)
        # z_g = F.adaptive_avg_pool2d(Y_g_mod, 1).view(Y_g_mod.size(0), -1)
        # z = torch.cat([z_l, z_g], dim=1)
        # weights = F.softmax(self.fusion_weight(z), dim=1)

        # # Weighted sum
        # Y = weights[:, 0].view(-1, 1, 1, 1) * Y_l_mod + weights[:, 1].view(-1, 1, 1, 1) * Y_g_mod

        #return Y
        return Y_l_mod, Y_g_mod




class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, enable_lfu=True):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.downsample = nn.Identity()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=False)
        )
        self.fu = FourierUnit(out_channels//2, out_channels//2)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels//2, out_channels//2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        target_size = x.shape[2:]  # (target_H, target_W)
        fu_out = self.fu(x)
        if self.enable_lfu:
            pooled = F.adaptive_avg_pool2d(x, (2, 2))
            lfu_out = self.lfu(pooled)
            lfu_out = F.interpolate(lfu_out, size=target_size, mode='nearest')
            if lfu_out.shape[2:] != target_size:
                lfu_out = center_crop(lfu_out, target_size[0], target_size[1])
        else:
            lfu_out = 0
        out = self.conv2(x + fu_out + lfu_out)
        return out

class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()
        assert stride in [1, 2]
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gout = ratio_gout
        self.convl2l = nn.Identity() if (in_cl == 0 or out_cl == 0) else nn.Conv2d(in_cl, out_cl,
                                                                                 kernel_size, stride,
                                                                                 padding, dilation,
                                                                                 groups, bias)
        self.convl2g = nn.Identity() if (in_cl == 0 or out_cg == 0) else nn.Conv2d(in_cl, out_cg,
                                                                                 kernel_size, stride,
                                                                                 padding, dilation,
                                                                                 groups, bias)
        self.convg2l = nn.Identity() if (in_cg == 0 or out_cl == 0) else nn.Conv2d(in_cg, out_cl,
                                                                                 kernel_size, stride,
                                                                                 padding, dilation,
                                                                                 groups, bias)
        self.convg2g = nn.Identity() if (in_cg == 0 or out_cg == 0) else SpectralTransform(in_cg, out_cg,
                                                                                           stride, enable_lfu)
    def forward(self, x):
        # Ensure x is a tuple (local, global)
        if not isinstance(x, tuple):
            x = (x, torch.zeros_like(x))
        x_l, x_g = x
        out_l_local = self.convl2l(x_l)
        out_l_global = self.convg2l(x_g)
        out_g_local = self.convl2g(x_l)
        out_g_global = self.convg2g(x_g)
        # For global branch: crop outputs to matching sizes before addition.
        if not isinstance(out_g_local, int) and not isinstance(out_g_global, int):
            out_g_local, out_g_global = match_size(out_g_local, out_g_global)
            out_g = out_g_local + out_g_global
        else:
            out_g = 0
        # For local branch: crop outputs to matching sizes before addition.
        if not isinstance(out_l_local, int) and not isinstance(out_l_global, int):
            out_l_local, out_l_global = match_size(out_l_local, out_l_global)
            out_l = out_l_local + out_l_global
        else:
            out_l = 0
        return out_l, out_g

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 enable_lfu=True):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu)
        self.dual_attn = DualFourierAttention(
            in_channels_local=int(out_channels*(1-ratio_gout)),
            in_channels_global=int(out_channels*ratio_gout)
        )
        self.bn_l = nn.Identity() if ratio_gout == 1 else norm_layer(int(out_channels*(1-ratio_gout)))
        self.bn_g = nn.Identity() if ratio_gout == 0 else norm_layer(int(out_channels*ratio_gout))
        self.act_l = nn.Identity() if ratio_gout == 1 else activation_layer(inplace=False)
        self.act_g = nn.Identity() if ratio_gout == 0 else activation_layer(inplace=False)
    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        x_l, x_g = self.dual_attn(x_l, x_g)

        return x_l, x_g
