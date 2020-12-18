"""
"""
import numpy as np 
import random
from eda import read_styles_csv
import pickle
from image_model import Model
from PIL import Image
from tqdm import tqdm
import os

DATA_PATH = 'data.pickle'
IMAGE_EMBEDDINGS_PATH = 'image_embeddings.pickle'

def read_pickle(pickle_file):
	with open(pickle_file, 'rb') as f:
		a = pickle.load(f)
	return a

def write_pickle(a, pickle_file):
	with open(pickle_file, 'wb') as f:
		pickle.dump(a, f)

def batch_embed(data):
    vision_model = Model()
    final_embeddings = [] 
    k=0
    batch_size = 32
    data_size = len(data)
    print(data_size)
    print("Apx No of batches = {}".format(data_size/batch_size))
    batch_num = 0
    while k< data_size:
        if batch_num%10 ==0:
            print(k)

        image_paths = data[k:k+batch_size]
        try:
            # print(image_paths)
            new_embeddings = list(vision_model(image_paths))
            final_embeddings+=new_embeddings
            k+=batch_size
            batch_num+=1
        except Exception as inst:
            print(inst)
            print("Invalid received")
            print("facing error for index {}, exiting".format(k))
            break
    return final_embeddings

def check_file(filename):
    try:
        dim = np.asarray(Image.open(filename)).shape[-1]
    except:
        dim = 1
    if dim!=3:
        return False
    return True

def read_image_folder(image_folder):
    image_list = [] 

    for i, file in tqdm(enumerate(os.listdir(image_folder))):
        full_path = os.path.join(image_folder, file)
        if not os.path.isdir(full_path):
            if check_file(full_path):
                image_list.append(full_path)
    
    return image_list


def embed_image_data():
    if not os.path.exists(DATA_PATH):        
        # read images from directory
        data = read_image_folder('images/')
        write_pickle(data, DATA_PATH)
    else:
        data = read_pickle(DATA_PATH)

    print(len(data))

    # embed images
    image_embeddings = batch_embed(data)
    # Pickelize and save
    write_pickle(image_embeddings, IMAGE_EMBEDDINGS_PATH)

def try_embed():
    model = Model()
    # list_of_image_embeddings = model(list_of_image_locations)

def main():
    embed_image_data()

if __name__ == '__main__':
    main()