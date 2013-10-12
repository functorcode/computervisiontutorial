from sklearn import svm
import numpy 
import pylab
from helper import visualize

X=numpy.array([[0,0],[1,1],[1,0],[0,1]])
Y=[0,1,1,1]
svc=svm.SVC(C=1.0,kernel='linear')
trainedsvm=svc.fit(X,Y)
visualize(trainedsvm,X,Y,[[-1.5,1.5],[-1.5,1.5]],100)
print trainedsvm.predict([0,0])
print trainedsvm.predict([1,0])
