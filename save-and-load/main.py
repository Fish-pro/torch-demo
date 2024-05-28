import torch
import torchvision.models as models

model = models.vgg16("IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")