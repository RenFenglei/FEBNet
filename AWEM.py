# ------------------------------------------------------------------------------
# Written by Fenglei Ren (renfenglei15@mails.ucas.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWaveletEnhancementModule(nn.Module):
    """
    Adaptive Wavelet Enhancement Module (AWEM)
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels

        # 低频语义增强分支 (处理 LL 分量)
        self.low_freq_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )  

        # 高频细节增强分支 (处理 LH, HL, HH 分量)
        self.high_freq_process = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=1, groups=1), # Channel interaction
            nn.Sigmoid() # 生成门控权重
        )

        # 3. 融合后的轻量级投影 (可选，用于平滑伪影)
        self.post_process = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def dwt_init(self, x):
        """
        离散小波变换 (Haar)
        x: [B, C, H, W]
        Returns: LL, (LH, HL, HH) stacked
        """
        # 对输入进行 Padding 以处理奇数分辨率
        h, w = x.size(2), x.size(3)
        pad_bottom = h % 2
        pad_right = w % 2
        if pad_bottom or pad_right:
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode='reflect')

        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        # Haar Wavelet Decomposition
        ll = x1 + x2 + x3 + x4 # Approximation (Low Freq)
        lh = x1 + x2 - x3 - x4 # Horizontal Detail
        hl = x1 - x2 + x3 - x4 # Vertical Detail
        hh = x1 - x2 - x3 + x4 # Diagonal Detail

        return ll, torch.cat([lh, hl, hh], dim=1)

    def idwt_init(self, ll, high_freqs):
        """
        逆离散小波变换 (Inverse Haar)
        """
        c = ll.shape[1]
        lh, hl, hh = torch.split(high_freqs, c, dim=1)

        x1 = (ll + lh + hl + hh) / 2
        x2 = (ll + lh - hl - hh) / 2
        x3 = (ll - lh + hl - hh) / 2
        x4 = (ll - lh - hl + hh) / 2

        # Interleave elements to restore original resolution
        # [B, C, H/2, W/2] -> [B, C, H, W]
        b, c, h, w = ll.shape
        x_out = torch.zeros(b, c, h * 2, w * 2, device=ll.device)

        x_out[:, :, 0::2, 0::2] = x1
        x_out[:, :, 1::2, 0::2] = x2
        x_out[:, :, 0::2, 1::2] = x3
        x_out[:, :, 1::2, 1::2] = x4

        return x_out

    def forward(self, x):
        shortcut = x

        # 频域分解 (Analysis)
        ll, high_freqs = self.dwt_init(x)

        # 频域独立增强 (Processing)
        # Branch A: Low Frequency
        # 对低频部分施加通道注意力，强化有用的语义通道
        ll_att = self.low_freq_att(ll)
        ll_enhanced = ll * ll_att

        # Branch B: High Frequency
        # 对高频部分施加空间门控，抑制噪声，锐化边缘
        high_mask = self.high_freq_process(high_freqs)
        high_enhanced = high_freqs * high_mask

        # 频域重建 (Synthesis)
        x_reconstructed = self.idwt_init(ll_enhanced, high_enhanced)

        # 处理 Padding 带来的尺寸不匹配
        if x_reconstructed.shape[2:] != x.shape[2:]:
            x_reconstructed = x_reconstructed[:, :, :x.shape[2], :x.shape[3]]

        # 4. 残差连接与后处理
        return self.post_process(x_reconstructed) + shortcut
    

if __name__ == '__main__':
    # 模拟语义分割中的特征图输入
    input_tensor = torch.randn(2, 64, 128, 128) 
    block = AdaptiveWaveletEnhancementModule(in_channels=64)
    output = block(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in block.parameters())}")