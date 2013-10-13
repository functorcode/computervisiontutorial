import cv2
import numpy
from sklearn import svm
from sklearn.externals import joblib
from helper import loadImages,getHog,detect,detectMultiScale,mergeRectangles,cropImages
from sklearn.metrics import confusion_matrix

#load images
positiveImages=loadImages('Pos.txt')
posDescriptors=getHog(positiveImages)
negativeImages=loadImages('Neg.txt')
negDescriptors=getHog(negativeImages)


#create training data
nPosSamples=posDescriptors.shape[0]
nNegSamples=negDescriptors.shape[0]

posLables=1.0*numpy.ones( (nPosSamples), dtype='float')  # not numpy.ones( (nPosSamples,1), dtype='float')
negLables=-1.0*numpy.ones( (nNegSamples), dtype='float')
lables=numpy.concatenate((posLables,negLables))
descriptors=numpy.concatenate((posDescriptors,negDescriptors))


#train
C=0.01
svc = svm.SVC(kernel='linear',tol=0.00000000001,probability=True, C=C)
print descriptors.shape, lables.shape
trained=svc.fit(descriptors,lables)

#test

predictedLabels=trained.predict(descriptors)
cm = confusion_matrix(lables, predictedLabels)
print(cm)


#load

#imgorig=cv2.imread('/home/juned/Code/computervisiontutorial/database/pascal/bottleOrignal/3.jpg')

#detect

detected=detectMultiScale(trained,imgorig,level=1)
detected=mergeRectangles(detected)


#display
for rect in detected:
	cv2.rectangle(imgorig, rect[0],rect[1],(0,255,0))
cv2.imshow("pedestrian",imgorig)
cv2.waitKey(0)
