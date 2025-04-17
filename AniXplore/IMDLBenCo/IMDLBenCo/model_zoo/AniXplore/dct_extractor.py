import torch
import torch.nn as nn
import torch.fft
import math
class DctFrequencyExtractor(nn.Module):
    def __init__(self, alpha=0.05):
        super(DctFrequencyExtractor, self).__init__()
        if alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        self.alpha = alpha
        self.dct_matrix_h = None
        self.dct_matrix_w = None

    def create_dct_matrix(self, N):
        n = torch.arange(N, dtype=torch.float32).reshape((1, N))
        k = torch.arange(N, dtype=torch.float32).reshape((N, 1))
        dct_matrix = torch.sqrt(torch.tensor(2.0 / N)) * torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0, :] = 1 / math.sqrt(N)
        return dct_matrix

    def dct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h, torch.matmul(x, self.dct_matrix_w.t()))

    def idct_2d(self, x):
        H, W = x.size(-2), x.size(-1)
        if self.dct_matrix_h is None or self.dct_matrix_h.size(0) != H:
            self.dct_matrix_h = self.create_dct_matrix(H).to(x.device)
        if self.dct_matrix_w is None or self.dct_matrix_w.size(0) != W:
            self.dct_matrix_w = self.create_dct_matrix(W).to(x.device)
        
        return torch.matmul(self.dct_matrix_h.t(), torch.matmul(x, self.dct_matrix_w))

    def high_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int(alpha * h), int(alpha * w)
        mask[:alpha_h, :alpha_w] = 0

        return x * mask

    def low_pass_filter(self, x, alpha):
        h, w = x.shape[-2:]
        mask = torch.ones(h, w, device=x.device)
        alpha_h, alpha_w = int((1.0-alpha) * h), int((1.0-alpha) * w)
        mask[-alpha_h:, -alpha_w:] = 0

        return x * mask

    def forward_high(self, x):
        xq = self.dct_2d(x)
        xq_high = self.high_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_high)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh

    def forward_low(self, x):
        xq = self.dct_2d(x)
        xq_low = self.low_pass_filter(xq, self.alpha)
        xh = self.idct_2d(xq_low)
        B = xh.shape[0]
        min_vals = xh.reshape(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = xh.reshape(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        xh = (xh - min_vals) / (max_vals - min_vals)
        return xh
    

if __name__ == "__main__":

    import cv2
    from torchvision import transforms

    img_bgr = cv2.imread("/home/fmg/chenyang/AI-edit-dataset/anime-example/SDXL_inpainting_10050_background.png", cv2.IMREAD_COLOR)
    # 将BGR转换为RGB（因为通常我们按RGB顺序处理）
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # 转换为 torch tensor，并调整 shape 为 (1, C, H, W)
    transform = transforms.ToTensor()  # 自动将像素值归一化到[0,1]
    img_tensor = transform(img_rgb).unsqueeze(0)  # (1, 3, H, W)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = HighDctFrequencyExtractor().to(device)
    img_tensor = img_tensor.to(device)

    high_feq = extractor(img_tensor)
    print(high_feq.shape)
    high_feq = (high_feq.squeeze().cpu().numpy() * 255).astype("uint8")
    print(high_feq.shape)
    print(high_feq)
    print(high_feq.min(), high_feq.max())
    high_feq = high_feq.transpose(1, 2, 0)
    cv2.imwrite("high_feq.png", high_feq)
    exit()

    # 获取 DWT 系数并保存子带图像  为什么报错啊 Invalid image height in IHDR libpng warning: Image height exceeds user limit in IHDR 宽度也是

    coeffs = extractor.dwt_2d(img_tensor)
    idwt_result = extractor.idwt_2d(coeffs)  # 重构的原始图像
    high_freq_coeffs = extractor.high_pass_filter(coeffs)
    high_freq_image = extractor.idwt_2d(high_freq_coeffs)  # 提取高频后的图像

    # 定义归一化函数（各自归一化到 [0,1] 范围）
    def normalize_tensor(tensor):
        B = tensor.shape[0]
        min_vals = tensor.view(B, -1).min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        max_vals = tensor.view(B, -1).max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        norm = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
        return norm

    idwt_result_norm = normalize_tensor(idwt_result)
    high_freq_image_norm = normalize_tensor(high_freq_image)

    save_path_hf = "hf.png"
    save_path_idwt = "idwt.png"
    cv2.imwrite(save_path_idwt, (idwt_result_norm.squeeze().cpu().numpy() * 255).astype("uint8"))
    cv2.imwrite(save_path_hf, (high_freq_image_norm.squeeze().cpu().numpy() * 255).astype("uint8"))