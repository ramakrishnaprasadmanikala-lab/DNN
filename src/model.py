import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------
#  Simple CNN Backbone for Image Encoding
# ----------------------------------------------------------
class ImageEncoder(nn.Module):
    """
    Lightweight CNN encoder to convert an image tensor into a feature vector.
    """
    def __init__(self, img_channels=3, embed_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 16, kernel_size=3, stride=2, padding=1),  # [B,16,112,112]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),            # [B,32,56,56]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),            # [B,64,28,28]
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),           # [B,128,14,14]
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))                                      # [B,128,1,1]
        )

        self.project = nn.Linear(128, embed_dim)

    def forward(self, x):
        features = self.encoder(x)             # [B,128,1,1]
        features = features.view(x.size(0), -1)  # [B,128]
        return self.project(features)          # [B,embed_dim]


# ----------------------------------------------------------
#  Text Encoder
# ----------------------------------------------------------
class TextEncoder(nn.Module):
    """
    Embeds text tokens + reduces them into a fixed vector.
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, tokens):
        embedded = self.embedding(tokens)                # [B,T,E]
        _, (h, _) = self.lstm(embedded)                  # h = [1,B,H]
        return h.squeeze(0)                              # [B,H]


# ----------------------------------------------------------
#  Baseline Fusion Model
# ----------------------------------------------------------
class BaselineModel(nn.Module):
    """
    Combines image + text representations and predicts next token distribution.
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)

        fusion_dim = embed_dim + hidden_dim
        self.fusion = nn.Linear(fusion_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, tokens):
        img_vec = self.image_encoder(images)   # [B,E]
        txt_vec = self.text_encoder(tokens)    # [B,H]

        combined = torch.cat([img_vec, txt_vec], dim=1)  # [B,E+H]
        fused = torch.relu(self.fusion(combined))        # [B,H]

        return self.classifier(fused)                    # [B,V]


# ----------------------------------------------------------
#  Cross-Modal Attention Fusion Model
# ----------------------------------------------------------
class AttentionEnhancedModel(nn.Module):
    """
    Introduces learnable attention between image and text features.
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_heads=4):
        super().__init__()

        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim)

        # Project both into same attention space
        self.img_proj = nn.Linear(embed_dim, hidden_dim)
        self.txt_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, tokens):
        img_vec = self.image_encoder(images)      # [B,E]
        txt_vec = self.text_encoder(tokens)       # [B,H]

        img_q = self.img_proj(img_vec)           # [B,H]
        txt_kv = self.txt_proj(txt_vec)          # [B,H]

        img_q = img_q.unsqueeze(1)               # [B,1,H]
        txt_kv = txt_kv.unsqueeze(1)             # [B,1,H]

        attn_out, _ = self.attention(img_q, txt_kv, txt_kv)

        attn_out = attn_out.squeeze(1)           # [B,H]

        return self.classifier(attn_out)         # [B,V]
