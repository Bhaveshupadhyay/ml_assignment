"""ONE-TIME online step: download ImageNet ResNet18 weights and save them into models/.

After this, the service never needs the network. In a real air-gapped deployment this
file (or the whole models/ dir) is what gets carried across on approved media.
"""

import hashlib

import torch
from torchvision import models

from app import config

config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
torch.save(net.state_dict(), config.BACKBONE_PATH)
digest = hashlib.sha256(config.BACKBONE_PATH.read_bytes()).hexdigest()
print(f"saved {config.BACKBONE_PATH} sha256={digest[:16]}")
