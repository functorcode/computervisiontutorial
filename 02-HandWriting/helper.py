from sklearn import datasets
import cv2
import pylab as pl
import numpy
from sklearn.cross_validation import ShuffleSplit

def loadImages(fileName,limit=-1):
	f=open(fileName)
	lines=f.readlines()
	imgList=[]
	counter=0
	for line in lines:
#		print "loading",line
		line=line.strip()

		if len(line)>0:
			if counter>limit and limit!=-1:
				break
			img=cv2.imread(line)
			if img != None:
				imgList.append(img)
				counter=counter+1
			#imgList.append(cv2.imread(line))
#			cv2.imshow("img",cv2.imread(line.strip()))
#			cv2.waitKey(1)
			
	return imgList

def getFeatures(imageList):
	if len(imageList) ==0:
		print "Error:Lenght of imageList can not be zero"
		return []

	features=numpy.ndarray((len(imageList),imageList[0].shape[0]*imageList[0].shape[1]),dtype='float')
	for i in range(len(imageList)):
		image=imageList[i]
		img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		features[i,:]=img.reshape(1,img.shape[0]*img.shape[1])
	return features

def train_test_split(features,labels, test_fraction=0.33, random_state=42):
	 indexes=ShuffleSplit(features.shape[0],n_iterations=1,test_fraction=0.33,random_state=42)
	
	 for train,test in indexes:
	 	trainIndex=train
	 	testIndex=test
	
	 trainFeatures=numpy.ndarray((len(trainIndex),features.shape[1]),dtype='float')
	 trainLabels=numpy.ndarray((len(trainIndex)),dtype='float')

	 testFeatures=numpy.ndarray((len(testIndex),features.shape[1]),dtype='float')
	 testLabel=numpy.ndarray((len(testIndex)),dtype='float')
	 counter=0

	 for train in trainIndex:
	 	trainFeatures[counter,:]=features[train,:]
	 	trainLabels[counter]=labels[train]
	 	counter=counter+1

	 counter=0
	 for test in testIndex:
	 	testFeatures[counter,:]=features[test,:]
	 	testLabel[counter]=labels[test]
	 	counter=counter+1
	 return trainFeatures,trainLabels,testFeatures,testLabel

def extractImages(outputFolder):
	digits = datasets.load_digits()

	imgDict={}

	for i in range(len(digits.images)):

		dig=digits.images[i]
		target=digits.target[i]
		img=numpy.zeros((8,8,1))

		for i in range(0,8):
			for j in range(0,8):
				img[i][j]=(dig[i][j] /16.0)*255.0
		imgDict.setdefault(target,[]).append(img)

	for key,Value in imgDict.items():
		counter=0
		for img in Value:
			cv2.imwrite(outputFolder+"/"+str(key)+"_"+str(counter)+".jpg",img)
			counter=counter+1
