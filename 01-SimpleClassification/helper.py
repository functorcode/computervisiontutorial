import numpy
import pylab

def visualize(trainedsvm,data,labels,axis_range,nSamplePoint):
    xx=numpy.linspace(axis_range[0][0],axis_range[0][1],nSamplePoint)
    yy=numpy.linspace(axis_range[1][0],axis_range[1][1],nSamplePoint)
    xx,yy=numpy.meshgrid(xx,yy)
    Z=trainedsvm.predict(numpy.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    pylab.contourf(xx,yy,Z)
    
    color_map = { 0: (0, 0, .9), 1: (1, 0, 0)}
    colors = [color_map[y] for y in labels]
    pylab.scatter(data[:, 0], data[:, 1], c=colors, cmap=pylab.cm.Paired)
    pylab.show()