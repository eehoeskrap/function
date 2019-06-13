import cv2


vidcap = cv2.VideoCapture('/home/seohee/Videos/HM_HandWave_025.mp4')

ret,image = cap.read()

count = 0


while ret:

	cv2.imwrite("./keypoint/frame%d.jpg" % count, image) 
     
	ret, image = cap.read()

	print('Read a new frame: ', ret)

	count += 1
