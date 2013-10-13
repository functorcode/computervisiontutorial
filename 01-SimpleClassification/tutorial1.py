from sklearn import svm
import numpy 
import pylab
from helper import visualize ,dataFromFile

obj1=dataFromFile('object1.txt')
obj2=dataFromFile('object2.txt')
label1= [1]*len(obj1)
label2=[0]*len(obj2)

trainingData=numpy.array( obj1+obj2)
labels=numpy.array(label1+label2)

svc=svm.SVC(C=1.0,kernel='linear')
trainedsvm=svc.fit(trainingData,labels)

visualize(trainedsvm,trainingData,labels,[[-1,25],[-1,25]],100)

#X=numpy.array([[0,0],[1,1],[1,0],[0,1]])
#Y=[0,1,1,1]
#trainedsvm=svc.fit(X,Y)
#print "True Label for [0,0] :" ,0, " Predicted :" ,trainedsvm.predict([0,0])
#print "True Label for [1,0:" ,1, " Predicted :" ,trainedsvm.predict([1,0])

#visualize(trainedsvm,X,Y,[[-1.5,1.5],[-1.5,1.5]],100)

#sample_weight=[2,1,1,1]