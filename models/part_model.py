import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
import timm
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class InceptionResnetBackbone(nn.Module):
	def __init__(self, args):
		super().__init__()

		pretrained = 'vggface2' if args.pretrained else None
		self.backbone = InceptionResnetV1(pretrained=pretrained)
		self.backbone.last_bn = nn.Identity()
		self.backbone.last_linear = nn.Linear(in_features=1792, out_features=512, bias=True)

	def forward(self, x):
		ret = self.backbone(x)

		return ret

class EyeEncoder(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.backbone = timm.create_model('resnet18', pretrained=args.pretrained)
		self.backbone.fc = nn.Linear(in_features=512, out_features=256, bias=True)

	def forward(self, x):
		ret = self.backbone(x)
		return ret

class PartModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.img_backbone = InceptionResnetBackbone(args)
		self.eye_backbone = EyeEncoder(args)

		self.fc = nn.Sequential(
			nn.Linear(1024, 512, bias=True),
			nn.GELU(),
			nn.Linear(512, 3, bias=True),
		)

	def forward(self, img, left_eye, right_eye):
		img_features = self.img_backbone(img)
		left_eye_features = self.eye_backbone(left_eye)
		right_eye_features = self.eye_backbone(right_eye)
		x = torch.cat((img_features, left_eye_features, right_eye_features), dim=1)
		x = self.fc(x)

		return x

class NewInceptionResnetBackbone(InceptionResnetV1):
	def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
		super().__init__(pretrained=pretrained, classify=classify, num_classes=num_classes, dropout_prob=dropout_prob, device=device)
	
	def forward(self, x):
		"""Calculate embeddings or logits given a batch of input image tensors.

		Arguments:
			x {torch.tensor} -- Batch of image tensors representing faces.

		Returns:
			torch.tensor -- Batch of embedding vectors or multinomial logits.
		"""
		x = self.conv2d_1a(x)
		x = self.conv2d_2a(x)
		x = self.conv2d_2b(x)
		x = self.maxpool_3a(x)
		x = self.conv2d_3b(x)
		x = self.conv2d_4a(x)
		x = self.conv2d_4b(x)
		x = self.repeat_1(x)
		x = self.mixed_6a(x)
		x = self.repeat_2(x)
		x = self.mixed_7a(x)
		x = self.repeat_3(x)
		x = self.block8(x)
		return x

class NewEyeEncoder(nn.Module):
	def __init__(self, args):
		super().__init__()

		self.backbone = timm.create_model('resnet18', pretrained=True)
		self.backbone.fc = nn.Identity()
		self.backbone.global_pool = nn.Identity()

	def forward(self, x):
		ret = self.backbone(x)
		return ret

class EyeAttention(nn.Module):
	def __init__(self, args, in_dim, activation):
		super().__init__()
		self.chanel_in = in_dim
		self.activation = activation
		
		self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
		self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
		self.gamma = nn.Parameter(torch.zeros(1))

		self.softmax  = nn.Softmax(dim=-1) #
	def forward(self,source,feat):
		"""
			inputs :
				source : input feature maps( B X C X W X H) 256,64,64
				driving : input feature maps( B X C X W X H) 256,64,64
			returns :
				out : self attention value + input feature 
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize,C,width ,height = source.size()
		proj_query  = self.activation(self.query_conv(source)).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) [bz,32,64,64]
		proj_key =  self.activation(self.key_conv(feat)).view(m_batchsize,-1,width*height) # B X C x (*W*H)
		energy =  torch.bmm(proj_query,proj_key) # transpose check
		attention = self.softmax(energy) # BX (N) X (N) 
		proj_value = self.activation(self.value_conv(feat)).view(m_batchsize,-1,width*height) # B X C X N

		out = torch.bmm(proj_value,attention.permute(0,2,1) )
		out = out.view(m_batchsize,C,width,height)
		out = self.gamma*out + feat

		return out,attention   

class PartModel2(nn.Module):
	def __init__(self, args):
		super().__init__()
		pretrained = 'vggface2' if args.pretrained else None
		self.img_backbone = NewInceptionResnetBackbone(pretrained=pretrained)
		self.eye_backbone = NewEyeEncoder(args)

		self.reshape1 = Rearrange('b c h w -> b c (h w)')

		self.upsample = nn.Sequential(
			Rearrange('b c n -> b n c'),
			nn.Linear(512, 1792),
			Rearrange('b n c -> b c n'),			
			nn.Linear(8, 25),
			Rearrange('b c (h w) -> b c h w', h=5, w=5)
		)

		self.eye_attention = EyeAttention(args, 1792, nn.ReLU())

		self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(0.6)
		self.last_linear = nn.Linear(1792, 3, bias=True)
	
	def forward(self, img, left_eye, right_eye):
		img_features = self.img_backbone(img)
		left_eye_features = self.eye_backbone(left_eye)
		right_eye_features = self.eye_backbone(right_eye)

		left_eye_features = self.reshape1(left_eye_features)
		right_eye_features = self.reshape1(right_eye_features)
		eye_features = torch.cat((left_eye_features, right_eye_features), dim=-1)

		eye_features = self.upsample(eye_features)

		x, _ = self.eye_attention(img_features, eye_features)

		x = self.avgpool_1a(x)
		x = self.dropout(x)
		x = self.last_linear(x.view(x.shape[0], -1))

		return x

class EyeModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.eye_backbone = EyeEncoder(args)

		self.fc = nn.Sequential(
			nn.Linear(512, 256, bias=True),
			nn.GELU(),
			nn.Linear(256, 3, bias=True),
		)

	def forward(self, img, left_eye, right_eye):
		left_eye_features = self.eye_backbone(left_eye)
		right_eye_features = self.eye_backbone(right_eye)
		x = torch.cat((left_eye_features, right_eye_features), dim=1)
		x = self.fc(x)

		return x

class PartModel3(nn.Module):
	def __init__(self, args):
		super().__init__()
		pretrained = 'vggface2' if args.pretrained else None
		self.img_backbone = NewInceptionResnetBackbone(pretrained=pretrained)
		self.eye_backbone = NewEyeEncoder(args)

		self.reshape = nn.Sequential(
			Rearrange('b c h w -> b (h w) c'),
			nn.Linear(512, 128),
			Rearrange('b n c -> b c n'),
		)
		self.upsample = nn.Linear(8, 64)

		self.reshape_img_features = nn.Sequential(
			Rearrange('b c h w -> b (h w) c'),
			nn.Linear(1792, 128),
			Rearrange('b n c -> b c n'),
			nn.Linear(25, 64)
		)

		encoder_layer = TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=64 * 4, dropout=0.1, batch_first=True, norm_first=True, activation='gelu') 
		self.transformer = TransformerEncoder(encoder_layer, num_layers=4)
		self.cls_token = nn.Parameter(torch.randn(1, 1, 64))
		
		self.pos_embed = nn.Parameter(torch.zeros(1, 257, 64))

		self.fc = nn.Linear(64, 3)

	def forward(self, img, left_eye, right_eye):
		img_features = self.img_backbone(img)
		img_features = self.reshape_img_features(img_features)
		left_eye_features = self.eye_backbone(left_eye)
		right_eye_features = self.eye_backbone(right_eye)

		left_eye_features = self.reshape(left_eye_features)
		right_eye_features = self.reshape(right_eye_features)
		eye_features = torch.cat((left_eye_features, right_eye_features), dim=-1)
		eye_features = self.upsample(eye_features)

		x = torch.cat((img_features, eye_features), dim=1)
		b, _, _ = x.shape
		cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
		x = torch.cat([cls_tokens, x], dim=1)

		x += self.pos_embed
		x = self.transformer(x)
		
		ret = x[:, 0]
		ret = self.fc(ret)
		
		return ret