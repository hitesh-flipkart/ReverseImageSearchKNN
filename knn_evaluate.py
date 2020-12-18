"""
Run on python 3.5
"""
from annoy import AnnoyIndex
import pickle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np 
from tqdm import tqdm
import random

DATA_PATH = 'data.pickle'
IMAGE_EMBEDDINGS_PATH = 'image_embeddings.pickle'

def read_pickle(pickle_file):
	with open(pickle_file, 'rb') as f:
		a = pickle.load(f)
	return a

def write_pickle(a, pickle_file):
	with open(pickle_file, 'wb') as f:
		pickle.dump(a, f)

def example_annoy():
	f = 40
	t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
	for i in range(1000):
		v = [random.gauss(0, 1) for z in range(f)]
		t.add_item(i, v)

	t.build(10) # 10 trees
	t.save('test.ann')

	u = AnnoyIndex(f, 'euclidean')
	u.load('test.ann') # super fast, will just mmap the file
	print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors

def prepare_annoy_indices(image_embeddings, annoy_filename='image_embeddings_annoy_indices.ann', num_trees= 10):
	embedding_size = len(image_embeddings[0])
	t = AnnoyIndex(embedding_size, 'euclidean')  # Length of item vector that will be indexed
	print("Adding embeddings one by one")
	for i in tqdm(range(len(image_embeddings))):
		t.add_item(i, image_embeddings[i])
	print("Building the tree")
	t.build(num_trees) # 10 trees
	print("Tree built")
	t.save(annoy_filename)
	print("Annoy indices saved at location: {}".format(annoy_filename))


def prepare_annoy_indices_train():
	print("Preparing annoy indices for train")
	data = read_pickle(DATA_PATH)
	image_embeddings = read_pickle(IMAGE_EMBEDDINGS_PATH)
	prepare_annoy_indices(image_embeddings, annoy_filename='image_embeddings_annoy_indices.ann', num_trees= 5000)


def inference():
	data = read_pickle(DATA_PATH)
	image_embeddings = read_pickle(IMAGE_EMBEDDINGS_PATH)
	embedding_size = len(image_embeddings[0])
	t = AnnoyIndex(embedding_size, 'euclidean')  # Length of item vector that will be indexed
	t.load('image_embeddings_annoy_indices.ann') # super fast, will just mmap the file
	return t

	for i in random.sample(range(len(image_embeddings)), 10):
		print("***********************************************")
		print("Original image location: {}".format(data[i]))
		results = t.get_nns_by_item(i, 5)
		print("Top 5 knn image ids: {}".format([data[k] for k in results]))

def main():
	prepare_annoy_indices_train()
	# inference()

if __name__ == '__main__':
	main()
		

