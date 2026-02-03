import torch
import torch.nn as nn
from transformers import AutoModel

from .config import MODEL_NAME, ID2LABEL


class HybridTransformerStylometric(nn.Module):
    def __init__(self, stylometric_dim: int = 100, num_labels: int = 3):
        super().__init__()

        # Transformer backbone
        self.transformer = AutoModel.from_pretrained(MODEL_NAME)

        hidden_size = self.transformer.config.hidden_size  # e.g. 768

        # Stylometric branch
        self.stylo_mlp = nn.Sequential(
            nn.Linear(stylometric_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Fusion + classifier
        fused_dim = hidden_size + 64
        self.classifier = nn.Linear(fused_dim, num_labels)

        self.id2label = ID2LABEL
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, stylometric_features=None, labels=None):
        # Transformer encoding
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        # Stylometric features (placeholder: if None, use zeros)
        if stylometric_features is None:
            batch_size = input_ids.size(0)
            device = input_ids.device
            stylometric_features = torch.zeros(batch_size, 100, device=device)

        stylo_repr = self.stylo_mlp(stylometric_features)

        # Concatenate
        fused = torch.cat([cls_embedding, stylo_repr], dim=1)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }
