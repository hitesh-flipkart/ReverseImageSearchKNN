import os
import numpy as np
import random
from tqdm import tqdm

data_path = '/shared/ayesha.s/fk_review_images'
filenames = [] 
for i, filename in tqdm(enumerate(os.listdir(data_path))):
    if i==30000:
        break
    filename = os.path.join(data_path, filename)
    if os.path.isdir(filename):
        continue
    filenames.append(filename)


images_sample = random.sample(filenames, 10000)
for image in tqdm(images_sample):
    os.system('cp {} images/'.format(image))
