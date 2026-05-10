# ------------------------------------------------------------------------------
# Written by Fenglei Ren (renfenglei15@mails.ucas.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(DAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm(branch_planes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((F.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[0])))
        x_list.append((self.process2((F.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[1]))))
        x_list.append(self.process3((F.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[2])))
        x_list.append(self.process4((F.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=algc)+x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out 
    
class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()
        bn_mom = 0.1
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        
        self.scale_process = nn.Sequential(
                                    BatchNorm(branch_planes*4, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes*4, branch_planes*4, kernel_size=3, padding=1, groups=4, bias=False),
                                    )

      
        self.compression = nn.Sequential(
                                    BatchNorm(branch_planes * 5, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        
        self.shortcut = nn.Sequential(
                                    BatchNorm(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )


    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                        mode='bilinear', align_corners=algc)+x_)
        
        scale_out = self.scale_process(torch.cat(scale_list, 1))
       
        out = self.compression(torch.cat([x_,scale_out], 1)) + self.shortcut(x)
        return out


class HPPM(nn.Module):
    """
    Hybrid Pyramid Pooling Module (HPPM)
    Innovation:
    1. Decouples Context Propagation (Sequential Add) from Feature Refinement (Parallel Conv).
    2. Maintains the 'Global-to-Local' information flow of DAPPM.
    3. Achieves the hardware efficiency of PAPPM.
    4. Adds a lightweight Channel Attention for adaptive scale fusion.
    """
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(HPPM, self).__init__()
        bn_mom = 0.1
        self.align_corners = False

        # === Multi-scale Pooling Branches (Same as DAPPM/PAPPM) ===
        # Scale 0: Original Resolution (1x1)
        self.scale0 = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        
        # Pooling Branches: Constructed in a loop for cleanliness
        # Parameters aligned with DAPPM: 5x5, 9x9, 17x17, Global
        self.pool_branches = nn.ModuleList()
        pool_sizes = [5, 9, 17]
        strides = [2, 4, 8]
        paddings = [2, 4, 8]
        
        for ks, st, pd in zip(pool_sizes, strides, paddings):
            self.pool_branches.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=ks, stride=st, padding=pd),
                BatchNorm(inplanes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
            ))
            
        # Global Branch (Adaptive)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        # === Parallel Refinement Modules ===
        # Unlike DAPPM, these are NOT inside the dependency loop.
        # They process the "fused" features in parallel.
        
        self.refine_modules = nn.ModuleList([
            nn.Sequential(
                BatchNorm(branch_planes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False)
            ) for _ in range(5) # 4 pooling + 1 identity
        ])
        
        '''
        self.refine_modules = nn.ModuleList([
            nn.Sequential(
                # 深度卷积 (Depthwise Conv)
                # groups=branch_planes 确保每个输入通道独立卷积
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, 
                          groups=branch_planes, bias=False),
                BatchNorm(branch_planes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                
                # 逐点卷积 (Pointwise Conv)
                # 使用 1x1 卷积将通道组合起来
                nn.Conv2d(branch_planes, branch_planes, kernel_size=1, bias=False),
                BatchNorm(branch_planes, momentum=bn_mom),
                nn.ReLU(inplace=True),
            ) for _ in range(5) # 5个分支
        ])
        '''

        # === Aggregation & Compression ===
        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )

        # === Shortcut ===
        self.shortcut = nn.Sequential(
            BatchNorm(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

        # === Innovation: Lightweight Scale-Attention ===
        # Reweights the 5 branches before compression
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(branch_planes * 5, branch_planes * 5 // 2, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5 // 2, branch_planes * 5, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    '''
    # coarse to fine
    def forward(self, x):
        height, width = x.shape[-2:]
        
        # Step 1: Extract features from all scales (Parallelizable on GPU)
        # s0 is finest, s_global is coarsest
        s0 = self.scale0(x)
        s1 = self.pool_branches[0](x)
        s2 = self.pool_branches[1](x)
        s3 = self.pool_branches[2](x)
        s_global = self.global_pool(x)
        
        raw_scales = [s_global, s3, s2, s1, s0] # Coarse to Fine order
        
        # Step 2: Lightweight Context Cascade (The "Hybrid" Part)
        # We propagate context from Coarse -> Fine using simple Addition.
        # No convolutions here! Only Upsample + Add.
        # This fixes DAPPM's latency bottleneck.
        fused_scales = []
        last_scale = None
        
        for s in raw_scales:
            # Resize current scale to original resolution
            s_up = F.interpolate(s, size=(height, width), mode='bilinear', align_corners=self.align_corners)
            
            if last_scale is not None:
                # Additive Fusion: Context flows from previous (coarser) scale
                current_fused = s_up + last_scale
            else:
                current_fused = s_up
            
            fused_scales.append(current_fused)
            last_scale = current_fused

        # Now fused_scales contains: [Global, Global+S3, Global+S3+S2, ...]
        # Note: The order in list is Coarse->Fine. Let's align with refine modules.
        # We usually want to concat in a specific order. Let's reverse to Fine->Coarse or keep as is.
        # DAPPM usually concats scale0...scale4. Let's stick to that order for consistency.
        # fused_scales is [Global, S3', S2', S1', S0'].
        # Let's reverse it to match DAPPM output order: [S0', S1', S2', S3', Global]
        fused_scales = fused_scales[::-1] 

        # Step 3: Parallel Refinement (High Parallelism)
        # Apply 3x3 Convs to all fused scales simultaneously
        out_list = []
        for i, feat in enumerate(fused_scales):
            out_list.append(self.refine_modules[i](feat))
            
        # Step 4: Concatenation & Attention
        cat_out = torch.cat(out_list, dim=1)
        
        # Apply Channel Attention to re-weight importance of different depths
        attn = self.scale_attn(cat_out)
        cat_out = cat_out * attn
        
        # Step 5: Final Compression & Shortcut
        out = self.compression(cat_out) + self.shortcut(x)
        
        return out
        '''


    # fine to coarse    
    def forward(self, x):
        height, width = x.shape[-2:]
        
        # --- Step 1: Parallel Feature Extraction ---
        # S0 (Fine) -> S1 -> S2 -> S3 -> S_global (Coarse)
        s0 = self.scale0(x)
        s1 = self.pool_branches[0](x)
        s2 = self.pool_branches[1](x)
        s3 = self.pool_branches[2](x)
        s_global = self.global_pool(x)
        
        # 定义 Fine-to-Coarse 顺序的尺度列表
        # S0 是最细，S_global 是最粗
        raw_scales = [s0, s1, s2, s3, s_global] 

        # --- Step 2: Fine-to-Coarse (F2C) Lightweight Cascade ---
        # 信息流：从 S0 开始，逐级向 S_global 累加
        fused_scales = []
        accumulated_sum = None # 使用 None 或 torch.zeros_like(s0) 初始化
        
        for s in raw_scales:
            # 1. Upsample current scale to original resolution
            s_up = F.interpolate(s, size=(height, width), 
                                 mode='bilinear', align_corners=self.align_corners)
            
            if accumulated_sum is None:
                # 初始化：第一个元素 (S0)
                current_fused = s_up
                accumulated_sum = current_fused
            else:
                # 累加：当前尺度 + 之前所有细尺度信息的总和
                current_fused = s_up + accumulated_sum 
                accumulated_sum = current_fused # 更新累加总和
            
            fused_scales.append(current_fused)

        # 此时 fused_scales 包含: [S0, S0+S1, S0+S1+S2, S0+S1+S2+S3, S0+S1+S2+S3+Sg]

        # --- Step 3: Parallel Refinement ---
        # 应用 3x3 卷积到所有融合后的尺度上（与原HPPM相同，并行执行）
        refined_list = []
        for i, feat in enumerate(fused_scales):
            refined_list.append(self.refine_modules[i](feat))
            
        # --- Step 4: Final Aggregation & Shortcut ---
        cat_out = torch.cat(refined_list, dim=1)
        
        # 应用尺度注意力
        attn = self.scale_attn(cat_out)
        cat_out = cat_out * attn
        
        out = self.compression(cat_out) + self.shortcut(x)
        
        return out



if __name__ == '__main__':
    # 模拟语义分割中的特征图输入
    input_tensor = torch.randn(2, 64*16, 16, 32) 
    block = HPPM(64*16,96,64*4)
    output = block(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in block.parameters())}")