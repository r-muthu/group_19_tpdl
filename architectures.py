import torch.nn as nn
import torchvision.models as models

class ViTB16(nn.Module):
	def __init__(self, num_classes=2):
		super(ViTB16, self).__init__()
		self.id = self.__class__.__name__
		self.model = models.vit_b_16(pretrained=True)
		self.model.heads = nn.Sequential(
			nn.Linear(self.model.hidden_dim, num_classes)
		)

	def forward(self, x):
		return self.model(x)


class ViTB32(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.vit_b_32(pretrained=True)
		self.model.heads = nn.Sequential(
			nn.Linear(self.model.hidden_dim, num_classes)
		)

	def forward(self, x):
		return self.model(x)

class ResNet50(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet50(pretrained=True)
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)

class ResNet152(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.resnet152(pretrained=True)
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

	def forward(self, x):
		return self.model(x)
	
class Densenet169(nn.Module):
	def __init__(self, num_classes=2):
		super().__init__()
		self.id = self.__class__.__name__
		self.model = models.densenet169(pretrained=True)
		self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

	def forward(self, x):
		return self.model(x)