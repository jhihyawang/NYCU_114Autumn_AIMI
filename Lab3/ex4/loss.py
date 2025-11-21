"""
損失函數列表:
1. Cross Entropy Loss (CE) - 標準交叉熵
2. Weighted Cross Entropy Loss (WCE) - 加權交叉熵
3. Focal Loss - 處理類別不平衡和難樣本
4. Label Smoothing Cross Entropy - 防止過擬合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import CLASS_NAMES

# Cross Entropy Loss
def get_ce_loss():
    """
    標準交叉熵損失
    
    優點:
    - 簡單直觀
    - 訓練穩定
    - 廣泛使用
    
    缺點:
    - 無法處理類別不平衡
    - 對簡單樣本浪費計算資源
    
    適用場景:
    - 數據集類別平衡
    - Baseline 實驗
    """
    return nn.CrossEntropyLoss()

#  Weighted Cross Entropy Loss (加權交叉熵)
def get_weighted_ce_loss(class_weights, device):
    """
    加權交叉熵損失
    為每個類別分配不同的權重
    
    Args:
        class_weights: 類別權重張量
        device: 計算設備 (cuda/cpu)
    
    權重計算公式:
        w_i = (總樣本數) / (類別數 × 第i類樣本數)
    
    優點:
    - 有效處理類別不平衡
    - 實現簡單
    - 訓練穩定
    
    缺點:
    - 權重固定，不自適應
    - 可能過度關注少數類別
    
    適用場景:
    - 類別不平衡數據集
    - 作為 Baseline
    """
    return nn.CrossEntropyLoss(weight=class_weights.to(device))


# Focal Loss
class FocalLoss(nn.Module):
    """
    Focal Loss 專門設計用於處理類別不平衡
    特別關注難分類樣本
    
    數學公式:
        L = -α_y × (1 - p_y)^γ × log(p_y)
    
    Args:
        alpha: 類別權重 (可選)
        gamma: focusing 參數，預設 2.0
               - 0: 等同於加權 CE
               - 2: 標準配置
               - 5: 極度關注難樣本
        reduction: 損失聚合方式 ('mean', 'sum', 'none')
    
    優點:
    - 自動降低簡單樣本的損失貢獻
    - 關注難分類樣本
    - 處理嚴重類別不平衡
    
    缺點:
    - 對超參數敏感
    - 訓練可能不穩定
    - 需要仔細調參
    
    適用場景:
    - 嚴重類別不平衡
    - 存在大量簡單樣本
    - 難樣本很重要
    
    參考文獻:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 類別權重
        self.gamma = gamma  # focusing 參數
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        前向傳播
        
        Args:
            inputs: 模型輸出 logits (batch_size, num_classes)
            targets: 真實標籤 (batch_size,)
        
        Returns:
            focal_loss: 計算的 Focal Loss
        """
        # 計算基礎交叉熵 (不聚合)
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none', 
            weight=self.alpha
        )
        
        # 計算預測機率
        pt = torch.exp(-ce_loss)
        
        # 計算 Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # 聚合
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Label Smoothing Cross Entropy 
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    通過平滑目標標籤來防止模型過度自信，減少過擬合
    
    數學公式:
        soft_target_i = {
            (1 - ε)     if i = y (正確類別)
            ε/(K-1)     otherwise
        }
        L = -Σ soft_target_i × log(p_i)
    
    Args:
        epsilon: 平滑參數，預設 0.1
                - 0.0: 無平滑（硬標籤）
                - 0.1: 標準配置
                - 0.2: 強平滑
    
    優點:
    - 防止過擬合
    - 提高泛化能力
    - 模型校準更好
    
    缺點:
    - 可能降低訓練準確率
    - 不直接處理類別不平衡
    
    適用場景:
    - 訓練集較小
    - 模型容易過擬合
    - 需要良好的機率校準
    
    參考文獻:
        Szegedy et al., "Rethinking the Inception Architecture", CVPR 2016
    """
    def __init__(self, epsilon=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, target):
        """
        前向傳播
        
        Args:
            preds: 模型輸出 logits (batch_size, num_classes)
            target: 真實標籤 (batch_size,)
        
        Returns:
            loss: Label Smoothing 損失
        """
        n_classes = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        
        # 計算平滑標籤
        with torch.no_grad():
            # 初始化為均勻分佈的小值
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.epsilon / (n_classes - 1))
            # 在正確類別位置填入較大的值
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.epsilon)
        
        # 計算損失
        loss = torch.mean(torch.sum(-true_dist * log_preds, dim=-1))
        return loss

# ==========================================
# Factory Function
# ==========================================
def get_criterion(args, device, class_weights=None):
    """
    根據 args 返回對應的損失函數
    """
    loss_type = args.loss_type.lower()

    # 若 user 開啟 use_class_weight flag，就用傳入的 weights 或預設的手動權重
    use_weights = args.use_class_weight
    
    cw = torch.tensor([1.0, 1.0, 4.0, 1.5], device=device)
    print(f"[Loss Info] Class Weights used: {cw.cpu().numpy()}")

    # Loss Type Routing
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    elif loss_type == "wce":
        return nn.CrossEntropyLoss(weight=cw)

    elif loss_type == "focal":
        print(f"[Loss Info] Focal alpha: {cw}")
        return FocalLoss(alpha=cw, gamma=args.focal_gamma)

    elif loss_type == "label_smooth":
        return LabelSmoothingCrossEntropy(epsilon=args.label_smooth_eps)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")