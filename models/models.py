import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import ViT
import timm
from torchvision.models import resnet18
from facenet_pytorch import InceptionResnetV1

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = nn.Identity()

    def forward(self, x):
        ret = self.backbone(x)

        return ret

class BaselineModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = resnet18(pretrained=args.pretrained)
        self.backbone.fc = nn.Linear(in_features=512, out_features=3, bias=True)

class EfficientNetModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('efficientnet_b3', pretrained=args.pretrained)
        self.backbone.classifier = nn.Linear(in_features=1536, out_features=3, bias=True)

class ConvNextModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('convnext_small', pretrained=args.pretrained)
        self.backbone.head.fc = nn.Linear(in_features=768, out_features=3, bias=True)

class ViTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('vit_small_patch16_224', pretrained=args.pretrained)
        self.backbone.head = nn.Linear(in_features=384, out_features=3, bias=True)

class SwinModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=args.pretrained)
        self.backbone.head = nn.Linear(in_features=768, out_features=3, bias=True)

class CaitModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('cait_s24_224', pretrained=args.pretrained)
        self.backbone.head = nn.Linear(in_features=384, out_features=3, bias=True)

class TwinsModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('twins_svt_small', pretrained=args.pretrained)
        self.backbone.head = nn.Linear(in_features=512, out_features=3, bias=True)

class InceptionResnetIM(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.backbone = timm.create_model('inception_resnet_v2', pretrained=args.pretrained)
        self.backbone.classif = nn.Linear(in_features=1536, out_features=3, bias=True)

class InceptionResnet(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        pretrained = args.pretraining_dataset if args.pretrained else None
        self.backbone = InceptionResnetV1(pretrained=pretrained)
        self.backbone.last_bn = nn.Identity()
        self.backbone.last_linear = nn.Linear(in_features=1792, out_features=3, bias=True)
