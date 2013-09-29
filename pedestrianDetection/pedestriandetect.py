import cv2
import numpy
from sklearn import svm
from sklearn.externals import joblib

def loadImages(fileName):
	f=open(fileName)
	lines=f.readlines()
	imgList=[]
	for line in lines:
#		print "loading",line
		line=line.strip()
		if len(line)>0:
			imgList.append(cv2.imread(line))
#			cv2.imshow("img",cv2.imread(line.strip()))
#			cv2.waitKey(1)
	return imgList
def getHog(imgList):
	hog = cv2.HOGDescriptor()
	descriptors=numpy.ndarray([len(imgList),hog.getDescriptorSize()],dtype='float')

	for i in range(len(imgList)):
		des=hog.compute(imgList[i])
		descriptors[i,:]=des.transpose()
	return descriptors



def detect(trainedsvm,img):
	rect=[]
	for y in range(0,img.shape[0],8):
		for x in range(0,img.shape[1],8):
			y1=y
			y2=y1+win_y
			x1=x
			x2=x1+win_x
			if(y2>img.shape[0] or x2 > img.shape[1]):
				continue
			tmpImg=img[y1:y2,x1:x2]
			des=getHog([tmpImg])
			result=trainedsvm.predict(des[0,:])
			if (result[0]==1.0) :
				rect.append([(x1,y1),(x2,y2)])
				#cv2.imshow("pedestrian",tmpImg)
				#cv2.rectangle(imgorig,(x1*4,y1*4),(x2*4,y2*4),(0,255,0))
				#cv2.waitKey(0)
	return rect

def detectMultiScale(trainedsvm,Image,level):
	detectedRect=[]
	tmpImg=Image
	scale=2
	for l in range(1,level+1):
		tmpImg=cv2.pyrDown(tmpImg)#,dstsize=(int(tmpImg.shape[0]/scale)+1,int(tmpImg.shape[1]/scale)+1))
		print tmpImg.shape
		rectangles=detect(trainedsvm,tmpImg)

		if(len(rectangles)>0):
			print l, scale
			for rect in rectangles:
				cv2.rectangle(tmpImg, rect[0],rect[1],(0,255,0))
				detectedRect.append([tuple( r*((scale**l)) for r in rect[0]),tuple( r*((scale**l)) for r in rect[1])])
		cv2.imshow("img",tmpImg)
		
		cv2.waitKey(0)
	return detectedRect
	

def mergeRectangles(rectList):
	rect=[]
	groupby_x=[]
	for r in rectList:
		rect.append([r[0][0],r[0][1],r[1][0],r[1][1]])
	sorted_x=rect #sorted(rect,key=lambda x:x[0])
	for s_x in sorted_x:
		isIntersect=False
		count=0
		for gp_x in groupby_x:
			for g_x in gp_x: 
				inter_left=max(g_x[0],s_x[0]) 
				inter_top= max(g_x[1],s_x[1])  
				inter_right=min(g_x[2],s_x[2])  
				inter_bottom=min(g_x[3],s_x[3]) 
				area1=(g_x[2]-g_x[0]) *(g_x[3]-g_x[1])
				area2= (s_x[2]-s_x[0]) *(s_x[3]-s_x[1])
				area=max(area1,area2)
				if (inter_right >inter_left and inter_bottom >inter_top ):
					inter_area= (inter_right-inter_left) * (inter_bottom-inter_top)
					if inter_area/(area*1.0) > 0.7:
						#print "Intersection found"
						isIntersect=True
					break
			if isIntersect==True:
				groupby_x[count].append(s_x)
				break
			count=count+1
		if isIntersect==False:
			
			groupby_x.append([s_x])

	grouped=[]
	for gp in groupby_x:
		print len(gp)
		avg_left=0
		avg_top=0
		avg_right=0
		avg_bottom=0
		count=0
		for rec in gp:
			avg_left=avg_left+rec[0]
			avg_top=avg_top+rec[1]
			avg_right=avg_right+rec[2]
			avg_bottom=avg_bottom+rec[3]
			count=count+1
		avg_left=int(avg_left/count)
		avg_top=int(avg_top/count)
		avg_right=int(avg_right/count)
		avg_bottom=int(avg_bottom/count)
		grouped.append([(avg_left,avg_top),(avg_right,avg_bottom)])
	return grouped
			


# detected=[[(64, 88), (128, 216)], [(72, 88), (136, 216)], [(112, 88), (176, 216)], [(120, 88), (184, 216)], [(16, 96), (80, 224)], [(56, 96), (120, 224)], [(64, 96), (128, 224)], [(72, 96), (136, 224)], [(112, 96), (176, 224)], [(120, 96), (184, 224)], [(16, 104), (80, 232)], [(24, 104), (88, 232)], [(64, 104), (128, 232)], [(112, 104), (176, 232)], [(120, 104), (184, 232)], [(16, 112), (80, 240)], [(16, 120), (80, 248)], [(16, 128), (80, 256)]]
# print len(detected)
# imgorig=cv2.imread('/home/juned/Code/pycon2013/pedestrianDetection/427px-642_Pedestrian_Facilities.jpg')
# detected=mergeRectangles(detected)
# print detected
# for rect in detected:
# 	cv2.rectangle(imgorig,tuple( e*2 for e in rect[0]),tuple( e*2 for e in rect[1]),(0,255,0))
# # detectMultiScale(imgorig,3)
# cv2.imshow("pedestrian",imgorig)
# cv2.waitKey(0)
# exit()

negativeImages=loadImages('neg.txt')
negDescriptors=getHog(negativeImages)
positiveImages=loadImages('pos.txt')
posDescriptors=getHog(positiveImages)
nPosSamples=posDescriptors.shape[0]
nNegSamples=negDescriptors.shape[0]

posLables=1.0*numpy.ones( (nPosSamples), dtype='float')  # not numpy.ones( (nPosSamples,1), dtype='float')
negLables=-1.0*numpy.ones( (nNegSamples), dtype='float')
lables=numpy.concatenate((posLables,negLables))
descriptors=numpy.concatenate((posDescriptors,negDescriptors))
C = 1.0  # SVM regularization parameter

svc = svm.SVC(kernel='linear',probability=True, C=C)
print descriptors.shape, lables.shape
trained=svc.fit(descriptors,lables)
#joblib.dump(trained,'trainedsvm.pkl')
#img=cv2.imread('/home/juned/Code/pycon2013/database/mit/pedestrians128x64/per00917.ppm')
#neg=cv2.imread('/home/juned/Code/pycon2013/database/negative/patches/451.jpg')
imgorig=cv2.imread('/home/juned/Code/pycon2013/pedestrianDetection/pedestrian_wave.JPG')
#des=getHog([img])
win_x=64
win_y=128
#img=cv2.pyrDown(imgorig)
#print img.shape
detected=detectMultiScale(trained,imgorig,level=2)
detected=mergeRectangles(detected)
print detected
for rect in detected:
	cv2.rectangle(imgorig, rect[0],rect[1],(0,255,0))
	#cv2.rectangle(imgorig,tuple( e*2 for e in rect[0]),tuple( e*2 for e in rect[1]),(0,255,0))
	#cv2.waitKey(0)
# for y in range(0,img.shape[0],8):
# 	for x in range(0,img.shape[1],8):
# 		y1=y
# 		y2=y1+win_y
# 		x1=x
# 		x2=x1+win_x
# 		if(y2>img.shape[0] or x2 > img.shape[1]):
# 			continue
# 		tmpImg=img[y1:y2,x1:x2]
# 		des=getHog([tmpImg])
# 		result=trained.predict(des[0,:])
# 		if (result[0]==1.0) :
# 			cv2.imshow("pedestrian",tmpImg)
# 			cv2.rectangle(imgorig,(x1*4,y1*4),(x2*4,y2*4),(0,255,0))
# 			cv2.waitKey(0)
cv2.imshow("pedestrian",imgorig)
cv2.waitKey(0)
#print trained.predict(des[0,:])
#print trained.predict(des[1,:])
#print trained.predict(descriptors[0,:])
#print trained.predict(descriptors[-1,:])
