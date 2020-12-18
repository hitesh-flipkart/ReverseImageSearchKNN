import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np

class Model(nn.Module):
	def __init__(self, params=None):
		super(Model, self).__init__()
		self.model1 = models.resnext101_32x8d(pretrained=True).cuda().eval()
		self.model2 = models.vgg19_bn(pretrained=True).cuda().eval()
		modules = list(self.model1.children())[:-1]
		self.model1=nn.Sequential(*modules)

		modules2 = list(self.model2.children())[:-1]
		self.model2=nn.Sequential(*modules2)

		self.image_loader = transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								std=[0.229, 0.224, 0.225])])

		
	def forward(self, images= []):
		# '''gets a list of input image paths and returns a numpy tensor of embedding'''
		list_images = []
		for file_name in images:
			list_images.append(self.image_loader(Image.open(file_name)).cuda())
		
		input_data = torch.stack(list_images, dim=0)
		# print(self.model2(input_data).reshape(len(images), -1, 1, 1).size(), self.model1(input_data).size())
		return torch.cat(( self.model2(input_data).reshape(len(images), -1, 1, 1) ,  self.model1(input_data)), 1).squeeze().cpu().data.numpy()




model = Model()
list_of_image_locations = ["/shared/saurabh.m/101_ObjectCategories/airplanes/image_0002.jpg", "/shared/saurabh.m/101_ObjectCategories/airplanes/image_0002.jpg"]
image_embeddings = model(list_of_image_locations)
# list_of_image_embeddings = torch.unbind(image_embeddings, dim =0)
# print(list_of_image_embeddings)
print(image_embeddings.shape)
print(np.sum(image_embeddings, axis=1))