import torch
from torch import nn
from torch.autograd import Function

class BaseModel(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()   
        self.pre_net = nn.Linear(input_dim, 256)

        self.post_net = nn.Linear(256, output_dim)
        
        self.activate = nn.ReLU()

    def forward(self, x, padding_mask=None):
        x = self.activate(self.pre_net(x))

        x = x * (1 - padding_mask.unsqueeze(-1).float())
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)  # Compute average
        
        x = self.post_net(x)
        return x



# 梯度反转操作
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# Domain Adversarial Neural Network
class EmotionDANN(nn.Module):
    def __init__(self, input_dim=768, num_emotions=7, num_domains=2):
        super(EmotionDANN, self).__init__()

        # --- 1. 特征提取器 (MLP Feature Extractor) ---
        # 目标：从 768 维降维到 128 维，提取共性特征
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), # 归一化有助于不同领域的数据分布对齐
            nn.ReLU(),
            nn.Dropout(0.5),     # 强力 Dropout，破坏特定的内容特征
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # --- 2. 情绪分类器 (Label Predictor) ---
        # 目标：正确分类情绪
        self.emotion_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions)
        )

        # --- 3. 领域判别器 (Domain Discriminator) ---
        # 目标：分辨是真实数据还是合成数据
        # 注意：这里前面要接 GRL 反转梯度！
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains) # 输出 2: [Real, Synth]
        )

    def forward(self, x, padding_mask,alpha=1.0):
        # 1. mask
        mask_float = 1 - padding_mask.float()
        x = x * mask_float.unsqueeze(-1)
        x_pooled = x.sum(dim=1) / mask_float.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        # 2. 提取特征
        features = self.feature_extractor(x_pooled)

        # 3. 情绪预测分支 (正常训练)
        emotion_pred = self.emotion_classifier(features)

        # 4. 领域判别分支 (对抗训练)
        # 特征先通过 GRL，再进入判别器
        # forward 时 reverse_features = features
        # backward 时 gradient 会取反
        reverse_features = GradientReversal.apply(features, alpha)
        domain_pred = self.domain_classifier(reverse_features)

        return emotion_pred, domain_pred


class LinearProbe(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, padding_mask):
        """
        x: [B, T, D]
        padding_mask: [B, T]  (1 = padding, 0 = valid)
        """
        # mask
        x = x * (1 - padding_mask.unsqueeze(-1).float())

        # mean pooling
        x = x.sum(dim=1) / (1 - padding_mask.float()
                            ).sum(dim=1, keepdim=True)

        # linear classification
        logits = self.fc(x)
        return logits

