from __future__ import print_function
import pandas
import math
import numpy as np
import sklearn
import skimage

from sklearn.cluster import KMeans

from skimage.io import imread
import matplotlib.pyplot as plt

def psnr(_src,_labels,_centers):
	res = 0
	i = 0
	M, N, n = _src.shape
	_l = M*N
	for i_row in _src:
		for i_col in i_row:
			for j in xrange(3):
				cur_label = _labels[i]				
				res += (i_col[j] - _centers[cur_label][j]*255)*(i_col[j] - _centers[cur_label][j]*255)
			i+=1	
		
	res = 20*math.log10(255) - 10*math.log10(res / (3.0*_l))
	return res

def show_image(_src,_labels,_centers):

	_dst = np.ndarray(shape=_src.shape,dtype=float)
	#print(_src.shape)
	M, N, n = _src.shape
	i = 0
	for i_row in xrange(M):		
		for i_col in xrange(N):
			cur_label = _labels[i]
			_dst[i_row][i_col]= _centers[cur_label]
			i+=1		
	plt.imshow(_dst)
	plt.show()

image = imread('parrots.jpg')
#print(image)

img_flt = skimage.img_as_float(image)

numberOfRows = len(img_flt)*len(img_flt[0])
img_matrix = []
i = 0
for i_row in img_flt:
	for i_col in i_row:
		#print(i_col)
		img_matrix.append(i_col)


for i in xrange(20):
	kmeans = KMeans(init='k-means++', random_state=241, n_clusters = i+1)
	kmeans.fit(img_matrix)
	#print(kmeans.labels_)
	#print(kmeans.cluster_centers_)
	res = psnr(image,kmeans.labels_,kmeans.cluster_centers_)
	show_image(image,kmeans.labels_,kmeans.cluster_centers_)
	print(i+1,res)
	if res > 20:
		f1 = open('data/611.txt','w')
		print(i+1, file=f1)
		break

	
