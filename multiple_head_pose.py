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



#print("files: ",files)
count = 0
for file in files:
    img = cv2.imread(file)
    landmarks = get_landmarks(img)
    size = img.shape
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
    #print(features)
    print(features[0])
    print(features[0][0])


    image_points = np.array([
                            (features[0][0],features[0][1]),     # Nose tip
                            (features[1][0],features[1][1]),     # Chin
                            (features[2][0],features[2][1]),     # Left eye left corner
                            (features[3][0],features[3][1]),     # Right eye right corne
                            (features[4][0],features[4][1]),     # Left Mouth corner
                            (features[5][0],features[5][1])      # Right mouth corner
                        ], dtype="double")


    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])


    # Camera internals

    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    print("Camera Matrix :\n {0}".format(camera_matrix))

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    print("Rotation Vector:\n {0}".format(rotation_vector))
    print("Translation Vector:\n {0}".format(translation_vector))


    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose


    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)


    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(img, p1, p2, (255,0,0), 2)


    # Display image
    cv2.imshow("Output", img);
    cv2.imwrite('head_pose_0'+str(count)+'.jpg',img)

    #cv2.line(image_with_landmarks,(56,99),(69,160),(0,255,255),2)
    #cv2.imwrite('image_with_landmarks'+str(count)+'.jpg',image_with_landmarks)
    count = count+1
    #print(landmarks)

    #print(landmarks[30],"   ",landmarks[8],"    ",landmarks[45],"   ",landmarks[36],"   ",landmarks[64],"   ",landmarks[48])
    cv2.waitKey(0)