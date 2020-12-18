"""
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
from collections import Counter
from tqdm import tqdm
import os

class Model(nn.Module):
    def __init__(self, params=None):
        super(Model, self).__init__()
        self.model = models.resnet152(pretrained=True).cuda().eval()
        modules=list(self.model.children())[:-1]
        self.model=nn.Sequential(*modules)

        self.image_loader = transforms.Compose([
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								std=[0.229, 0.224, 0.225])])

    def forward(self, images= []):
        list_images = []
        for file_name in images:
            list_images.append(self.image_loader(Image.open(file_name)).cuda())
        
        input_data = torch.stack(list_images, dim=0)
        return self.model(input_data).squeeze().data.cpu().data.numpy()


# model = Model()
# list_of_image_locations = ["/shared/saurabh.m/101_ObjectCategories/airplanes/image_0002.jpg", "/shared/saurabh.m/101_ObjectCategories/airplanes/image_0002.jpg"]

# list_of_image_locations = ["images/21379.jpg", "images/39386.jpg"]
# image_embeddings = model(list_of_image_locations)
# # list_of_image_embeddings = torch.unbind(image_embeddings, dim =0)
# # print(list_of_image_embeddings)
# print(image_embeddings.shape)

# sizes = [] 
# for file_name in tqdm(os.listdir('images')):
#     file_name = os.path.join('images', file_name)
#     if os.path.isdir(file_name):
#         continue
#     sizes.append(np.asarray(Image.open(file_name)).shape[-1])
# print(Counter(sizes))
