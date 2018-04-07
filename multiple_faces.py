import argparse
import imutils
import dlib
import cv2
import numpy as np
# from matplotlib import pyplot as plt

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    rects = detector(im,1)
    print(len(rects))
    #if len(rects)>1:
    #   raise TooManyFaces
    if len(rects)==0:
        raise NoFaces

    return np.matrix([[p.x,p.y] for p in predictor(im,rects[0]).parts()])

def annotate_landmarks(im,landmarks):
    im = im.copy()
    for idx,point in enumerate(landmarks):
        pos = (point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
        cv2.circle(im,pos,3,color=(0,255,255))
    
    return im 


# detect faces in the grayscale image
rects = detector(gray, 1)
#print(len(rects))

fname = args["image"].split('/')[-1]
name, ext = fname.split('.')

# To get names of saved images
files = []
# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    (x, y, w, h) = rect_to_bb(rect)
    #print(i, x, y, w, h)

    fname = '{}_{}.{}'.format(name, i, ext)
    #print(fname)
    files.append(fname)
    # clone the original image so we can draw on it, then
    # display the name of the face part on the image
    clone = image.copy()
    cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 1)
    startX = x
    startY = y - 15 if y - 15 > 15 else y + 15
    cv2.putText(clone, str(i), (startX, startY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    roi = image[y:y + h, x:x + w]

    cv2.imshow("ROI", roi)
    cv2.imwrite(fname, roi)
    
    cv2.imshow("Image", clone)
    cv2.waitKey(0)



print("files: ",files)
count = 0
for file in files:
    img = cv2.imread(file)
    landmarks = get_landmarks(img)
    #image_with_landmarks = annotate_landmarks(img,landmarks)
    #cv2.imshow("Result",image_with_landmarks)
    features=[]
    features.append(landmarks[30])
    features.append(landmarks[8])
    features.append(landmarks[45])
    features.append(landmarks[36])
    features.append(landmarks[64])
    features.append(landmarks[48])
    features = np.squeeze(np.asarray(features))
    print(features)
    print(features[0])
    #cv2.line(image_with_landmarks,(56,99),(69,160),(0,255,255),2)
    #cv2.imwrite('image_with_landmarks'+str(count)+'.jpg',image_with_landmarks)
    count = count+1
    #print(landmarks)

    #print(landmarks[30],"   ",landmarks[8],"    ",landmarks[45],"   ",landmarks[36],"   ",landmarks[64],"   ",landmarks[48])
    cv2.waitKey(0)