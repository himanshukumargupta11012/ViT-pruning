from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        alpha: Balancing factor for class imbalance (0 < alpha < 1)
        gamma: Focusing parameter (higher = more focus on hard samples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs, targets):
        # Compute standard Binary Cross-Entropy Loss
        bce_loss = nn.BCELoss(reduction='none')(probs, targets)
        
        pt = probs * targets + (1 - probs) * (1 - targets)  # p_t = p for y=1, (1-p) for y=0
        
        # Apply Focal Loss scaling factor
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        
        # Apply class balancing weight (alpha)
        loss = self.alpha * targets * loss + (1 - self.alpha) * (1 - targets) * loss

        return loss.mean()
        