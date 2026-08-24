import torch
import torch.nn as nn
from transformers import AutoModel


class LanguageClassifier(nn.Module):
    """
    Multilingual Transformer model for language classification.
    """

    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        num_classes=12,
        dropout=0.1
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)

        hidden_size = self.transformer.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)

        return logits
