import cv2
import numpy
from sklearn import svm
from sklearn.externals import joblib

win_x=64
win_y=128
def loadImages(fileName):
	f=open(fileName)
	lines=f.readlines()
	imgList=[]
	for line in lines:
#		print "loading",line
		line=line.strip()
		if len(line)>0:
			img=cv2.imread(line)
			if img != None:
				imgList.append(img)
#			cv2.imshow("img",cv2.imread(line.strip()))
#			cv2.waitKey(1)
	return imgList


def getHog(imgList):
	hog = cv2.HOGDescriptor(_winSize=(win_x,win_y), _blockSize=(16,16), _blockStride=(8,8), _cellSize=(8,8), _nbins=9)
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
			result=trainedsvm.predict_proba(des[0,:])

			if (result[0][1]>0.8) :
				rect.append([(x1,y1),(x2,y2)])
				#cv2.imshow("pedestrian",tmpImg)
				#cv2.rectangle(imgorig,(x1*4,y1*4),(x2*4,y2*4),(0,255,0))
				#cv2.waitKey(0)
	return rect

def detectMultiScale(trainedsvm,Image,level):
	detectedRect=[]
	tmpImg=Image
	scale=2
#	rectangles=detect(trainedsvm,tmpImg)
#	if (len(rectangles)>0):
#		for rect in rectangles:
#			detectedRect.append([tuple( r*(1) for r in rect[0]),tuple( r*(1) for r in rect[1])])

	for l in range(1,level+1):
		tmpImg=cv2.pyrDown(tmpImg)#,dstsize=(int(tmpImg.shape[0]/scale)+1,int(tmpImg.shape[1]/scale)+1))
	#	print tmpImg.shape
		rectangles=detect(trainedsvm,tmpImg)

		if(len(rectangles)>0):
			#print l, scale
			for rect in rectangles:
				cv2.rectangle(tmpImg, rect[0],rect[1],(0,255,0))
				detectedRect.append([tuple( r*((scale**l)) for r in rect[0]),tuple( r*((scale**l)) for r in rect[1])])
		#cv2.imshow("img",tmpImg)
		
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
		#print len(gp)
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



def cropImages(imgFile,outFolder):
#imgFile="/home/juned/Code/pycon2013/database/negative/image_001115.jpg"
#outFolder="/home/juned/Code/pycon2013/database/negative/patches"
	img=cv2.imread(imgFile);
#	win_x=64
#	win_y=128
	tmpImg=img[0:128,0:64]
	counter=0
	for y in range(0,img.shape[0],64):
		for x in range(0,img.shape[1],64):
			y1=y
			y2=y1+win_y
			x1=x
			x2=x1+win_x
			if(y2>img.shape[0] or x2 > img.shape[1]):
				continue
			counter=counter+1
			tmpImg=img[y1:y2,x1:x2]
			cv2.imwrite(outFolder+str(counter)+".jpg",tmpImg)
				