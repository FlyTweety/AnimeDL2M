import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import math

# -------------------------------
# 前向 DWT：基于 Daubechies 4 小波
# -------------------------------
class DaubechiesDWT(nn.Module):
    def __init__(self):
        super(DaubechiesDWT, self).__init__()
        # db4 的低通滤波器系数
        db4 = [0.2303778133088964,
               0.7148465705529154,
               0.6308807679298587,
              -0.027983769416859854,
              -0.18703481171888114,
               0.030841381835986965,
               0.032883011666982945,
              -0.010597401785069032]
        # 将系数转换为 tensor
        h = torch.tensor(db4, dtype=torch.float32)  # 低通滤波器
        # 高通滤波器 g[n] = (-1)**n * h[7-n]
        g = torch.tensor([(-1)**n * db4[7 - n] for n in range(8)], dtype=torch.float32)
        
        # 构造 2D 滤波器（通过外积）
        # LL 分量
        kernel_LL = torch.ger(h, h)   # outer product, shape (8,8)
        # LH 分量：水平方向高通，垂直方向低通
        kernel_LH = torch.ger(h, g)
        # HL 分量：垂直方向高通，水平方向低通
        kernel_HL = torch.ger(g, h)
        # HH 分量：高通滤波器外积
        kernel_HH = torch.ger(g, g)
        
        # 将四个滤波器堆叠成一个 tensor，形状 (4, 1, 8, 8)
        filters = torch.stack([kernel_LL, kernel_LH, kernel_HL, kernel_HH], dim=0)
        filters = filters.unsqueeze(1)  # shape: (4,1,8,8)
        self.register_buffer('base_filter', filters)
        # 设定卷积的 padding
        self.pad = 3  # 对于 kernel_size=8，通常取 (8-1)//2 = 3 可使输出尺寸为原尺寸的一半（适用于偶数尺寸输入）

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 针对每个通道复制滤波器，得到 shape (4*C, 1, 8, 8)
        weight = self.base_filter.repeat(C, 1, 1, 1)
        # 使用 groups 卷积，每个通道独立处理
        coeffs = F.conv2d(x, weight, stride=2, padding=self.pad, groups=C)  # shape (B, 4*C, H_out, W_out)
        # 重构为 (B, C, 4, H_out, W_out)
        H_out = coeffs.shape[-2]
        W_out = coeffs.shape[-1]
        coeffs = coeffs.view(B, C, 4, H_out, W_out)
        # 将每个通道的系数组织为 (LL, (LH, HL, HH))
        coeffs_list = []
        for b in range(B):
            sample_coeffs = []
            for c in range(C):
                a = coeffs[b, c, 0:1, :, :]  # LL 分量
                b_coef = coeffs[b, c, 1:2, :, :]  # LH 分量
                c_coef = coeffs[b, c, 2:3, :, :]  # HL 分量
                d_coef = coeffs[b, c, 3:4, :, :]  # HH 分量
                sample_coeffs.append((a, (b_coef, c_coef, d_coef)))
            coeffs_list.append(sample_coeffs)
        return coeffs_list

# -------------------------------
# 逆变换 IDWT：基于 Daubechies 4 小波重构
# -------------------------------
class DaubechiesIDWT(nn.Module):
    def __init__(self):
        super(DaubechiesIDWT, self).__init__()
        # db4 低通滤波器系数
        db4 = [0.2303778133088964,
               0.7148465705529154,
               0.6308807679298587,
              -0.027983769416859854,
              -0.18703481171888114,
               0.030841381835986965,
               0.032883011666982945,
              -0.010597401785069032]
        h = torch.tensor(db4, dtype=torch.float32)
        g = torch.tensor([(-1)**n * db4[7 - n] for n in range(8)], dtype=torch.float32)
        # 重构滤波器为：翻转原滤波器序列
        h_rev = torch.flip(h, dims=[0])
        g_rev = torch.flip(g, dims=[0])
        # 构造重构用的 2D 滤波器
        kernel_LL = torch.ger(h_rev, h_rev)
        kernel_LH = torch.ger(h_rev, g_rev)
        kernel_HL = torch.ger(g_rev, h_rev)
        kernel_HH = torch.ger(g_rev, g_rev)
        # 堆叠为 shape (4, 1, 8, 8)
        filters = torch.stack([kernel_LL, kernel_LH, kernel_HL, kernel_HH], dim=0)
        filters = filters.unsqueeze(1)  # (4,1,8,8)
        self.register_buffer('synthesis_filter', filters)
        self.pad = 3  # 与前向变换使用相同的 padding

    def forward(self, coeffs_tensor, output_size):
        """
        coeffs_tensor: (B, 4, H_out, W_out) 针对单通道
        output_size: (H, W) 重构图像尺寸
        """
        # 利用 conv_transpose2d 进行上采样重构
        recon = F.conv_transpose2d(coeffs_tensor, self.synthesis_filter, stride=2, padding=self.pad)
        # 若重构尺寸略大，则裁剪到 output_size
        H, W = output_size
        recon = recon[:, :, :H, :W]
        return recon

# -------------------------------
# 高频特征提取器：利用 DWT 和 IDWT
# -------------------------------
class DwtFrequencyExtractor(nn.Module):
    def __init__(self):
        super(DwtFrequencyExtractor, self).__init__()
        self.dwt = DaubechiesDWT()
        self.idwt_module = DaubechiesIDWT()

    def dwt_2d(self, x):
        """
        对输入 x (B, C, H, W) 进行 DWT，返回嵌套列表：
        外层列表长度为 B，每个元素为长度为 C 的列表，每个元素为 (LL, (LH, HL, HH))
        """
        return self.dwt(x)

    def idwt_2d(self, coeffs_list):
        """
        对每个样本每个通道的系数进行逆变换，返回形状 (B, C, H, W) 的重构图像
        """
        B = len(coeffs_list)
        C = len(coeffs_list[0])
        recons = []
        for b in range(B):
            channels_recon = []
            for c in range(C):
                a, details = coeffs_list[b][c]
                b_coef, c_coef, d_coef = details
                # 拼接为 shape (1, 4, H_out, W_out)
                coeffs_tensor = torch.cat([a, b_coef, c_coef, d_coef], dim=0).unsqueeze(0)
                H_out, W_out = a.shape[-2:]  # 变换后尺寸
                # 原始尺寸为 (H_out*2, W_out*2)
                output_size = (H_out * 2, W_out * 2)
                recon = self.idwt_module(coeffs_tensor, output_size)  # (1,1,H,W)
                channels_recon.append(recon[0, 0].unsqueeze(0))
            sample_recon = torch.cat(channels_recon, dim=0)  # (C, H, W)
            recons.append(sample_recon.unsqueeze(0))
        recons = torch.cat(recons, dim=0)  # (B, C, H, W)
        return recons

    def high_pass_filter(self, coeffs_list):
        """
        对每个样本每个通道，将低频部分（LL）置零，仅保留高频细节
        """
        filtered_list = []
        for sample in coeffs_list:
            filtered_sample = []
            for (a, details) in sample:
                b_coef, c_coef, d_coef = details
                a_hp = torch.zeros_like(a)
                filtered_sample.append((a_hp, (b_coef, c_coef, d_coef)))
            filtered_list.append(filtered_sample)
        return filtered_list

    def forward(self, x):
        """
        1. 对输入图像进行 DWT 得到系数
        2. 将低频部分置零（保留高频）
        3. 利用 IDWT 重构图像，得到高频特征图
        """
        coeffs_list = self.dwt_2d(x)
        high_freq_coeffs = self.high_pass_filter(coeffs_list)
        xh = self.idwt_2d(high_freq_coeffs)
        # 将结果归一化到 [0,1]
        B = xh.shape[0]
        min_vals = xh.view(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals + 1e-8)
        return xh

# -------------------------------
# 测试代码
# -------------------------------
if __name__ == "__main__":
    from torchvision import transforms

    # 读取图像（请修改为您自己的图像路径）
    img_bgr = cv2.imread("/home/fmg/chenyang/AI-edit-dataset/anime-example/SDXL_inpainting_10050_background.png", cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("无法读取图像，请检查图像路径。")
    # 将 BGR 转换为 RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 转换为 torch tensor，形状 (1, C, H, W)，并归一化到 [0,1]
    transform = transforms.ToTensor()
    img_tensor = transform(img_rgb).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = HighDwtFrequencyExtractor().to(device)
    img_tensor = img_tensor.to(device)

    # 提取高频特征图
    high_freq = extractor(img_tensor)
    print("高频特征图形状：", high_freq.shape)
    
    # 转换为 NumPy 格式并调整通道顺序（C, H, W -> H, W, C）
    high_freq_np = (high_freq.squeeze().cpu().numpy() * 255).astype("uint8")
    # 若图像为多通道，则进行转置
    if high_freq_np.ndim == 3:
        high_freq_np = high_freq_np.transpose(1, 2, 0)
    # 保存结果图像
    cv2.imwrite("high_freq_image.png", high_freq_np)

    # # 显示结果（可选）
    # plt.imshow(high_freq_np)
    # plt.title("High Frequency Features (db4)")
    # plt.axis("off")
    # plt.show()
