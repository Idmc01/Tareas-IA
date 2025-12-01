import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Partial(nn.Module):
    def __init__(
        self,
        layers,
        embedding_dim,
        num_classes,
        hidden_dim,
        dropout=0.3,
    ):
        super().__init__()

        base_model = models.resnet18(weights=None)

        modules = {
            "conv1": nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool,
            ),
            "conv2_x": base_model.layer1,
            "conv3_x": base_model.layer2,
            "conv4_x": base_model.layer3,  # not used unless you enable it
            "conv5_x": base_model.layer4,  # not used unless you enable it
        }

        selected = []
        for name in layers:
            if name not in modules:
                raise ValueError(f"Layer {name} no existe en ResNet18 partial")
            selected.append(modules[name])

        self.feature_extractor = nn.Sequential(*selected)

        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_out = self.feature_extractor(dummy_input)
        flatten_dim = dummy_out.numel()

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        embedding = self.embedding(features)
        logits = self.classifier(embedding)
        return logits, embedding


def build_resnet18_partial(
    layers,
    embedding_dim,
    num_classes,
    hidden_dim,
    dropout,
):
    return ResNet18Partial(
        layers=layers,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )