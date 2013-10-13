import cv2
import numpy
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from helper import loadImages,getHog,detect,detectMultiScale,mergeRectangles,cropImages
			
#cropImages('/home/juned/Code/computervisiontutorial/database/negative/image_001115.jpg','/home/juned/Code/computervisiontutorial/database/negative/64x128/')
#exit()

#pedestrian images
positiveImages=loadImages('pos.txt')
posDescriptors=getHog(positiveImages)

#random patches
negativeImages=loadImages('neg.txt')
negDescriptors=getHog(negativeImages)


#prepare data
nPosSamples=posDescriptors.shape[0]
nNegSamples=negDescriptors.shape[0]
posLables=1.0*numpy.ones( (nPosSamples), dtype='float')  # not numpy.ones( (nPosSamples,1), dtype='float')
negLables=-1.0*numpy.ones( (nNegSamples), dtype='float')
lables=numpy.concatenate((posLables,negLables))
descriptors=numpy.concatenate((posDescriptors,negDescriptors))



C = 0.1  # SVM regularization parameter

svc = svm.SVC(kernel='linear',tol=0.000000001,probability=True, C=C)
#print descriptors.shape, lables.shape
trained=svc.fit(descriptors,lables)
#joblib.dump(trained,'trainedsvm.pkl')
predictedLabels=trained.predict(descriptors)

cm = confusion_matrix(lables, predictedLabels)
print(cm)


#load image

imgorig=cv2.imread('/home/juned/Code/computervisiontutorial/database/pedestrian/testImages/3.jpg' )


#detection

detected=detectMultiScale(trained,imgorig,level=1)
detected=mergeRectangles(detected)


#display

for rect in detected:
	cv2.rectangle(imgorig, rect[0],rect[1],(0,255,0))
	
cv2.imshow("pedestrian",imgorig)
cv2.waitKey(0)

