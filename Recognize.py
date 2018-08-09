import numpy as np
import cv2
import os

Class_1 = []
Class_2 = []
Class_3 = []
Class_4 = []
Class_5 = []
Class_6 = []

def find_descriptors(img,sift):
	_ , des = sift.detectAndCompute(img,None)
	return des
	       
        
def load_benchmark(sift):

	global Class_1
	global Class_2
	global Class_3
	global Class_4
	global Class_5
	global Class_6
		
	file_names = os.listdir("Benchmark")
	for file_name in file_names:
		img = cv2.imread("Benchmark/"+file_name,0)
		img = cv2.resize(img,(100,100), interpolation = cv2.INTER_CUBIC)
		file_name = file_name.split('.')[0]
		fl = file_name.split('_')
		temp = fl[1] + fl[2] + fl[3]
		if temp == "11f":
			Class_1.append(find_descriptors(img,sift))
		elif temp == "12f":
			Class_2.append(find_descriptors(img,sift))
		elif temp == "21f":
			Class_3.append(find_descriptors(img,sift))
		elif temp == "22f":
			Class_4.append(find_descriptors(img,sift))
		elif temp == "51f":
			Class_5.append(find_descriptors(img,sift))
		elif temp == "52f":
			Class_6.append(find_descriptors(img,sift))

def match (desc1 , desc2 , flann):
	matches = flann.knnMatch(desc1,desc2,k=2)
	matchesMask = [[0,0] for i in xrange(len(matches))]
	cnt = 0
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.7*n.distance:
		matchesMask[i]=[1,0]
		cnt += 1
	return cnt
	
def cross(img , sift):
	result = []
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   	
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	img_desc = find_descriptors(img , sift)
	temp = 0
	for item in Class_1:
		temp += match(img_desc , item , flann)
	temp = temp*1.0/len(Class_1)
	result.append(temp)		
	
	temp = 0
	for item in Class_2:
		temp += match(img_desc , item , flann)
	temp = temp*1.0/len(Class_2)
	result.append(temp)		

	temp = 0
	for item in Class_3:
		temp += match(img_desc , item , flann)
	temp = temp*1.0/len(Class_3)
	result.append(temp)		
	
	temp = 0
	for item in Class_4:
		temp += match(img_desc , item , flann)
	temp = temp*1.0/len(Class_4)
	result.append(temp)		

	temp = 0
	for item in Class_5:
		temp += match(img_desc , item , flann)
	temp = temp*1.0/len(Class_5)
	result.append(temp)		

	temp = 0
	for item in Class_6:
		temp += match(img_desc , item , flann)
	temp = temp*1.0/len(Class_6)
	result.append(temp)		
	
	return result	
		
def Scrape_Images():
	sift = cv2.xfeatures2d.SIFT_create()
	load_benchmark(sift)
	for file_name in (os.listdir("Detected_Images")):
		img = cv2.imread("Detected_Images/"+file_name,0)
		result = cross(img , sift)
		print file_name , np.argmax(result) + 1
	
	        
Scrape_Images()

