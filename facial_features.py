import cv2
import dlib
import numpy as np 

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

class TooManyFaces(Exception):
	pass

class NoFaces(Exception):
	pass

def get_landmarks(im):
	rects = detector(im,1)
	print(len(rects))
	#if len(rects)>1:
	#	raise TooManyFaces
	if len(rects)==0:
		raise NoFaces

	return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])

def annotate_landmarks(im,landmarks):
	im = im.copy()
	for idx,point in enumerate(landmarks):
		pos = (point[0,0],point[0,1])
		cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
		cv2.circle(im,pos,3,color=(0,255,255))
	
	#print(landmarks[30],"	",landmarks[8],"	",landmarks[45],"	",landmarks[36],"	",landmarks[64],"	",landmarks[48])
	return im 

image = cv2.imread('headPose1.jpg')
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image,landmarks)
print(landmarks[30],"	",landmarks[8],"	",landmarks[45],"	",landmarks[36],"	",landmarks[64],"	",landmarks[48])
cv2.imshow("Result",image_with_landmarks)
cv2.line(image_with_landmarks,(361,387),(185,427),(0,255,255),2)
cv2.imwrite('head_pose.jpg',image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()









