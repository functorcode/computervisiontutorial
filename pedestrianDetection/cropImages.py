import cv2
imgFile="/home/juned/Code/pycon2013/database/negative/image_001115.jpg"
outFolder="/home/juned/Code/pycon2013/database/negative/patches"
img=cv2.imread(imgFile);
win_x=64
win_y=128
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

