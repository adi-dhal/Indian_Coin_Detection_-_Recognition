import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import sys

def crop_circle(x,y,r,cnt,img):
	
	name = input("Response ")
	
	if name == 1:
		return
	else:
	
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		crop = img[int(y)-int(r):int(y)+int(r),int(x)-int(r):int(x)+int(r)]
		mask = np.zeros((int(r)+int(r),int(r)+int(r)))
		cv2.circle(mask,(int(r),int(r)),int(r),1,thickness=-1)
	
		print mask.shape
		print crop.shape
		if mask.shape != crop.shape:
			return
		final = cv2.multiply(mask,crop.astype(float))
		'''
		final = cv2.bitwise_and(img[int(x)-int(r):int(x)+int(r),int(y)-int(r):int(y)+int(r)], img[int(x)-int(r):int(x)+int(r),int(y)-int(r):int(y)+int(r)], mask=mask)
		'''
	
		cv2.imwrite("crop_1_1_b_"+str(cnt)+".jpg",final)

def pre_process(img):
	resize = cv2.resize(img,None,fx = 0.2,fy = 0.2, interpolation = cv2.INTER_CUBIC)

	lab = cv2.cvtColor(resize, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l)
	limg = cv2.merge((cl,a,b))
	new = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

	new = cv2.pyrMeanShiftFiltering(new, 21, 51)

	gray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

	x,y = gray.shape
	print x,y
	decision = np.sum(gray)*1.0/(x*y)

	if decision > 127: 
		thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	else:
		thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	return thresh , gray , resize
	
def Detect_Coins(thresh,cnt,gray,resize):	
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)

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
		cv2.circle(gray, (int(x), int(y)), int(r), (0, 255, 0), 2)
		cv2.imshow('detected circles',gray)
		cv2.waitKey(0)
		crop_circle(x,y,r,cnt,resize)
		cv2.destroyAllWindows()


#################Driver###################################
for i in range(int(sys.argv[1])):
	
	img = cv2.imread(str(i+1)+'.jpg')

	thresh , gray , resize = pre_process(img)
	
	Detect_Coins(thresh,i+1,gray,resize)


