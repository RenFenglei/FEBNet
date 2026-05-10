# ------------------------------------------------------------------------------
# Written by Fenglei Ren (renfenglei15@mails.ucas.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class FCBlock(nn.Module):
    
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x

class TripleStripAttentionGating(nn.Module):
    """
    WeightGenerator
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.row_fc = FCBlock(channels, reduction)
        self.col_fc = FCBlock(channels, reduction)
        self.cha_fc = FCBlock(channels, reduction)

    def forward(self, x):
        # 行条带注意力
        row = self.row_fc(torch.mean(x, dim=2, keepdim=True)) # [B, C, 1, W]
        
        # 列条带注意力
        col = self.col_fc(torch.mean(x, dim=3, keepdim=True)) # [B, C, H, 1]
        
        # 通道注意力
        cha = self.cha_fc(F.adaptive_avg_pool2d(x, 1))        # [B, C, 1, 1]
        
        # 4. 广播累加并生成权重
        attn_weight = torch.sigmoid(row + col + cha)          # [B, C, H, W]
        return attn_weight

class BilateralAdaptiveFusionModul(nn.Module):
    """
    Feature Fusion
    """
    def __init__(self, dim, reduction=4):
        super(BilateralAdaptiveFusionModul, self).__init__()
        self.weight_generator = TripleStripAttentionGating(dim, reduction)
        
        # 融合后的特征投影层，用于特征平滑和通道对齐
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        """
        x: 语义分支特征 (Semantic Context)
        y: 细节分支特征 (Spatial Details)
        """
        # 初始特征相加，聚合双分支的初步信息
        initial = x + y
        
        # 将聚合特征送入 TSAM 生成三维（行、列、通道）融合权重
        attn = self.weight_generator(initial)
        
        # 动态特征融合 (互补式门控机制)
        # attn 倾向于 1 的区域使用语义特征 x
        # attn 倾向于 0 的区域使用细节特征 y
        fusion_feat = x * attn + y * (1 - attn)
        
        # 残差连接增强信息流动，并进行最终的卷积投影
        result = initial + fusion_feat
        result = self.proj(result)
        
        return result
    


if __name__ == '__main__':
    # 模拟语义分割中的特征图输入
    x = torch.randn(2, 64, 128, 128)
    y = torch.randn(2, 64, 128, 128)
    block = BilateralAdaptiveFusionModul(dim = 64)
    output = block(x,y)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in block.parameters())}")