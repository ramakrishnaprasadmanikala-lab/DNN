import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class Trainer:
    """
    Simple training pipeline for your visual storytelling models.
    """

    def __init__(self, model, train_loader, val_loader, vocab_pad_idx, lr=1e-4, device="cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab_pad_idx)

    # ------------------------------
    # Training loop (one epoch)
    # ------------------------------
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_loader, desc="Training"):
            images = batch["images"].to(self.device)     # [B,3,224,224]
            tokens = batch["tokens"].to(self.device)     # [B,T]

            # Target is last token of caption (predict next-word / summary)
            target = tokens[:, -1]  # [B]

            self.optimizer.zero_grad()
            logits = self.model(images, tokens)          # [B, vocab]
            loss = self.loss_fn(logits, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # ------------------------------
    # Validation loop
    # ------------------------------
    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch["images"].to(self.device)
                tokens = batch["tokens"].to(self.device)

                target = tokens[:, -1]

                logits = self.model(images, tokens)
                loss = self.loss_fn(logits, target)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    # ------------------------------
    # Full training procedure
    # ------------------------------
    def fit(self, epochs=5):
        """
        Train and validate across N epochs, printing loss.
        """
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch}/{epochs} --> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        return self.model
