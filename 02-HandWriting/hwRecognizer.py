from sklearn import svm

from helper import loadImages,getFeatures,train_test_split
from sklearn.metrics import confusion_matrix
import numpy
import pylab as pl
import cv2
#load possitive
posImage=loadImages('1.txt')

nPosSamples=len(posImage)

#load negagive images
negImages=[]
negImgFileList=['0.txt','2.txt','3.txt','4.txt','5.txt','6.txt','7.txt','8.txt','9.txt']
limit=20
for negfile in negImgFileList:
	negImages=negImages+loadImages(negfile,limit)
nNegSamples=len(negImages)

#prepare dataset


#positive
posFeatures=getFeatures(posImage)
posLabels=1.0* numpy.ones((nPosSamples),dtype='float')

#negatives
negFeatures=getFeatures(negImages)
negLabels=-1.0* numpy.ones((nNegSamples),dtype='float')

#create train and test set
allFeatures=numpy.concatenate((posFeatures,negFeatures))
allLabels=numpy.concatenate((posLabels,negLabels))

trainingFeatures, trainingLabels, testingFeatures, testingLabels = train_test_split(allFeatures,allLabels , test_fraction=0.33, random_state=42)

#train
svc=svm.SVC(C=1,kernel='linear')
trainedSVM=svc.fit(trainingFeatures,trainingLabels)


#prediction
print "TrueClass: 1, Predicted: ", trainedSVM.predict(posFeatures[0,:])
print "TrueClass: -1, Predicted: ", trainedSVM.predict(negFeatures[0,:])
print "TrueClass: ",testingLabels[0]," ,Predicted: ", trainedSVM.predict(testingFeatures[0,:])



#confusion matrix
preidcitonLabels=trainedSVM.predict(trainingFeatures)

cm = confusion_matrix(trainingLabels, preidcitonLabels)
print(cm)
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True labelIndex')
pl.xlabel('Predicted label')
pl.show()



