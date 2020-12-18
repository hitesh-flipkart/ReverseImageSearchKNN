import pandas as pd
import numpy as np 
from collections import defaultdict, Counter


def read_styles_csv(filename, print_info=False):
	data = defaultdict(list)
	id_to_name = {}
	num_columns = 0
	with open(filename, 'r') as f:
		for i, line in enumerate(f):
			line = line.strip().split(',')
			if i==0:
				for k, elem in enumerate(line):
					id_to_name[k] = elem
				num_columns = len(line)
			else:
				if len(line)!=num_columns:
					continue
				for k, elem in enumerate(line):
					data[id_to_name[k]].append(elem)
	
	if print_info:
		print("Column names: {}".format(','.join(data.keys())))
		print("No of images: {}".format(len(data['id'])))

		for column_name in ['gender','masterCategory','subCategory','articleType','baseColour','season','year','usage']:
			print("*****************************************")
			print("Column name : {}".format(column_name))
			print(Counter(data[column_name]))
			
	return data
	
if __name__ == '__main__':
	read_styles_csv('styles.csv', print_info=True)