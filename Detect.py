import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import sys
import os
import glob

#from Recognize import Scrape_Images
def crop_circle(x,y,r,img,cnt):
	if int(r) < 25 or int(r) > 60:
		return cnt , 0
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	crop = img[int(y)-int(r):int(y)+int(r),int(x)-int(r):int(x)+int(r)]
	mask = np.zeros((int(r)+int(r),int(r)+int(r)))
	cv2.circle(mask,(int(r),int(r)),int(r),1,thickness=-1)
	if mask.shape != crop.shape:
		return cnt ,0
	final = cv2.multiply(mask,crop.astype(float))
	cv2.imwrite("Detected_Images/"+str(cnt)+".jpg",final)
	cnt += 1
	return cnt , 1

def pre_process(img):
	new = cv2.resize(img,None,fx = 0.2,fy = 0.2, interpolation = cv2.INTER_CUBIC)

	lab = cv2.cvtColor(new, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)
	limg = cv2.merge((cl,a,b))
	new = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	
	filtered = cv2.medianBlur(new,17)
	
	filtered = cv2.pyrMeanShiftFiltering(filtered, 41, 51)

	gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
		
	x,y = gray.shape
	
	decision = np.sum(gray)*1.0/(x*y)
	

	if decision > 127: 
		thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	else:
		thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		
	return thresh , gray , new , filtered
	
def Detect_Coins(thresh , gray , new):	
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)
	cnt = 0
	for label in np.unique(labels):
		if label == 0:
			continue
		mask = np.zeros(gray.shape, dtype="uint8")
		mask[labels == label] = 255
	 
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		c = max(cnts, key=cv2.contourArea)
	 
		# draw a circle enclosing the object
		((x, y), r) = cv2.minEnclosingCircle(c)
		cnt , res = crop_circle(x,y,r,new,cnt)
		if res == 1:
			cv2.circle(new, (int(x), int(y)), int(r), (0, 255, 0), 2)
	print cnt
	return new



#################Driver###################################

files = glob.glob("Detected_Images/*")
for f in files:
	os.remove(f)
img = cv2.imread(sys.argv[1]+'.jpg')

thresh , gray , new ,filtered= pre_process(img)
new = Detect_Coins(thresh , gray , new)

cv2.imshow('detected circles',new)
cv2.imwrite('result/'+str(sys.argv[1])+".jpg",new)
cv2.waitKey(0)
cv2.destroyAllWindows()


