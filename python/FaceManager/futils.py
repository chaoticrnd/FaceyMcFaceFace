#!/usr/bin/python

"""
Utils for FaceyMcFaceFace
@Author: Matt Murray
"""
import sys, os, glob, json, math, random, itertools
import numpy as np
import cv2


def process_face(detector, sp, facerec, img, calcDescriptor=False, shouldGaze=False):

    dets = detector(img, 1)
    results = []

    #fail well
    if len(dets) <= 0:
        return None

    # Now process each face we found.
    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        #get landmarks
        landmarks = get_landmarks(d, shape)
        #fail well if can't get landmarks
        if landmarks == None:
            return None

        headPose = None
        if shouldGaze:
            headPose = calc_pose_estimation(img, landmarks['labeled'])
            print("headPose: ", headPose)

        face_descriptor = []
        if calcDescriptor:
            face_descriptor = facerec.compute_face_descriptor(img, shape)
        
        faceData = {
            'faceDescriptor':list(face_descriptor),
            'bounds':{
                "left":d.left(),
                "top": d.top(),
                "right": d.top(),
                "bottom": d.bottom()
            },
            'headAngle':landmarks['headAngle'],
            'landmarks':landmarks['labeled'],
            'headPose':headPose
        }

        results.append({
            'descriptor':face_descriptor,
            'faceData':faceData,
            'landmarkData':landmarks
        })

    return results[0] #only return first face found if more than one for some odd reason

def get_landmarks(d, shape):
    #make a dict to store the face landmarks
    #with labels for the SDK in unity
    landmarksMap = {
        "0":"pupilLeft",
        "1":"pupilRight"
    }

    labeledPoints = {}
    i=0
    xlist = []
    ylist = []
    allList = []
    for pnt in shape.parts():
        xlist.append(float(pnt.x))
        ylist.append(float(pnt.y))
        
        allList.append(float(pnt.x))
        allList.append(float(pnt.y))

        labeledPoints[i] = (float(pnt.x), float(pnt.y))
        i+=1

    #get nose angle and vectorize landmarks
    xmean = np.mean(xlist) #Find both coordinates of centre of gravity
    ymean = np.mean(ylist)
    xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
    ycentral = [(y-ymean) for y in ylist]

    if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
        anglenose = 0
    else:
        anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi) #point 29 is the tip of the nose, point 26 is the top of the nose brigde

    if anglenose < 0: #Get offset by finding how the nose brigde should be rotated to become perpendicular to the horizontal plane
        anglenose += 90
    else:
        anglenose -= 90

    landmark_vectors = []
    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
        landmark_vectors.append(x) #Add the coordinates relative to the centre of gravity
        landmark_vectors.append(y)

        #Get the euclidean distance between each point
        #and the centre point (the vector length)
        meannp = np.asarray((ymean,xmean))
        coornp = np.asarray((z,w))
        dist = np.linalg.norm(coornp-meannp)
        landmark_vectors.append(dist)

        #Get the angle the vector describes relative to the image,
        #corrected for the offset that the nosebrigde has when
        #the face is not perfectly horizontal
        anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
        landmark_vectors.append(anglerelative)

    return {
        "labeled":labeledPoints,
        "vectorized":landmark_vectors,
        "xList":xlist,
        "yList":ylist,
        "all":allList,
        "headAngle": anglenose
    }

def calc_pose_estimation(im, landmarks):
    """
    ref: http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    figure out the pitch roll and yaw of a face
    points needed: tip of nose, chin, left and right
    outer eye corners, and left/right outer lip corners
    @param: size | shape object
    @param: list | face landmarks as list of tuples
    @return: dict of translation_vector and rotation_vector
    """
    size = im.shape
    # Nose tip, Chin, Left eye left corner, Right eye right corner, Left Mouth corner, Right mouth corner
    facepointindicies = [33,8,36,45,48,54] 
    neededCoords = []
    for fi in facepointindicies:
        neededCoords.append(landmarks[fi])

    #2D image points. If you change the image, you need to change vector
    image_points = np.array(neededCoords, dtype="double")
     
    # 3D model points. use this generic one for now
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])
    
    # Camera internals, generic for now too
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = "double"
        )
     
    #print "Camera Matrix :\n {0}".format(camera_matrix)
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    #do the magic here with slovePnP
    hp = {"headRotation":None, "headTranslation":None}
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)
    
    print("success: ", success)
    if success:
        #print "Rotation Vector:\n {0}".format(rotation_vector)
        #print "Translation Vector:\n {0}".format(translation_vector)
        hp["headRotation"] = rotation_vector
        hp["headTranslation"] = translation_vector

    return hp
    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    #(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    #for p in image_points:
    #    cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
     
    #make it pinnochio
    #p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    #p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    #cv2.line(im, p1, p2, (255,0,0), 2)

    # Display image
    #cv2.imshow("Output", im)
    #cv2.waitKey(0)

def data_uri_to_cv2_img(uri):
    """
    change base64 string image into cv2image
    """
    encoded_data = uri
    try:
        if not uri.index(',') == 0:
            encoded_data = uri.split(',')[1]
    except:
        print("erp")
    nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def data_to_img(inputData):
    """
    determine if input is file path
    or base64 and convert to image regardless
    """
    image = None
    if os.path.isfile(os.path.abspath(inputData)):
        print("this is a file: " + inputData)
        inputData = os.path.abspath(inputData)
        image = cv2.imread(inputData) #open image
    else:
        #assume it's base64
        image = data_uri_to_cv2_img(inputData)
    return image


def print_progress(current, total):
    print("progress: {0}%".format((float(current)/ total)*100))


