

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example shows how to use dlib's face recognition tool.  This tool maps
#   an image of a human face to a 128 dimensional vector space where images of
#   the same person are near to each other and images from different people are
#   far apart.  Therefore, you can perform face recognition by mapping faces to
#   the 128D space and then checking if their Euclidean distance is small
#   enough. 
#
#   When using a distance threshold of 0.6, the dlib model obtains an accuracy
#   of 99.38% on the standard LFW face recognition benchmark, which is
#   comparable to other state-of-the-art methods for face recognition as of
#   February 2017. This accuracy means that, when presented with a pair of face
#   images, the tool will correctly identify if the pair belongs to the same
#   person or is from different people 99.38% of the time.
#
#   Finally, for an in-depth discussion of how dlib's tool works you should
#   refer to the C++ example program dnn_face_recognition_ex.cpp and the
#   attendant documentation referenced therein.
#
#
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  This code will also use CUDA if you have CUDA and cuDNN
#   installed.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io
from scipy.spatial import distance as dist
import json
import argparse

#./face_recognition.py ../models/shape_predictor_68_face_landmarks.dat ../models/dlib_face_recognition_resnet_model_v1.dat ../data/faces
"""
You can download a trained facial shape predictor and recognition model from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
"""


predictor_path = os.path.abspath("../models/shape_predictor_68_face_landmarks.dat")
face_rec_model_path = os.path.abspath("../models/dlib_face_recognition_resnet_model_v1.dat")


print(predictor_path)
print(face_rec_model_path)

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.

print("loading models")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


def processAll(faces_folder_path):
    """
    iterate through all images in a dir and find faces then show landmarks
    """
    allImages = glob.glob(os.path.join(faces_folder_path, "*.jpg"))
    #print(allImages)
    # Now process all the images
    for f in allImages:
        print("Processing file: {}".format(f))
        img = io.imread(f)

        win.clear_overlay()
        win.set_image(img)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        # Now process each face we found.
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            # Draw the face landmarks on the screen so we can see what face is currently being processed.
            win.clear_overlay()
            win.add_overlay(d)
            win.add_overlay(shape)

            # Compute the 128D vector that describes the face in img identified by
            # shape.  In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people.  He we just print
            # the vector to the screen.
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            #print(face_descriptor)

            
            # It should also be noted that you can also call this function like this:
            #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100)
            # The version of the call without the 100 gets 99.13% accuracy on LFW
            # while the version with 100 gets 99.38%.  However, the 100 makes the
            # call 100x slower to execute, so choose whatever version you like.  To
            # explain a little, the 3rd argument tells the code how many times to
            # jitter/resample the image.  When you set it to 100 it executes the
            # face descriptor extraction 100 times on slightly modified versions of
            # the face and returns the average result.  You could also pick a more
            # middle value, such as 10, which is only 10x slower but still gets an
            # LFW accuracy of 99.3%.


            dlib.hit_enter_to_continue()


def train_faces(facesPath):
    pass

def pre_process_faces(facesPath):
    """
    @param: face images dir
    process all images and save data for each one
    """
    #iterate over all images in path and save the ladmarks detected
    allImages = glob.glob(os.path.join(facesPath, "*.jpg"))
    #print(allImages)
    # Now process all the images
    for f in allImages:
        pre_process_face(f)

def pre_process_face(f):
    """
    @param: face image path
    load a face, get landmarks and features, save to disk
    @return: face descriptors  = 128D vector
    """
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img, 1)

    # Now process each face we found.
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = sp(img, d)
        #get the landmarks
        landmarks = {}
        l=0
        for pnt in shape.parts():
            landmarks[l] = (pnt.x, pnt.y)
            l+=1

        face_descriptor = facerec.compute_face_descriptor(img, shape)
        #print(face_descriptor)
        faceDataPath = f.replace('.jpg', '.json')
        print("saving results to: {0}".format(faceDataPath))
        faceData = {
            'faceDescriptor':list(face_descriptor),
            'rect':{
                "left":d.left(),
                "top": d.top(),
                "right": d.top(),
                "bottom": d.bottom()
            },
            'landmarks':landmarks
        }
        save_features(faceDataPath, faceData)

        return face_descriptor

def save_features(pth, data):
    """
    @param: path to file
    @param: data of face, bounds, descriptors, and landmarks
    """
    with open(pth, 'w') as fdp:
        fdp.write(json.dumps(data))

def load_features(pth):
    """
    @param: file path of data
    @return: previously calculated descriptors
    """
    d = None
    with open(pth, 'r') as f:
        d = json.loads(f.read())

    return d['faceDescriptor']

def compare_two_faces(faceOne, faceTwo):
    """
    given 2 image paths, load them, find landmarks then compare euclidean
    distance to see if they are similar
    """

    #assume the descriptors have already
    #been calculated for the images but check to be safe

    dataPathA = faceOne.replace('.jpg', '.json')
    dataPathB = faceTwo.replace('.jpg', '.json')
    aData = None
    bData = None

    #see if the files are there
    if not os.path.isfile(dataPathA):
        aData = pre_process_face(faceOne)
    else:
        aData = load_features(dataPathA)

    if not os.path.isfile(dataPathB):
        bData = pre_process_face(faceTwo)
    else:
        bData = load_features(dataPathB)

    isMatch = distancePasses(aData, bData)

    if isMatch:
        print ( "These are the same people!")
    else:
        print ("these are not the same people")


def distancePasses(distA, distB, _max=0.59):
    distBetween = dist.euclidean(distA, distB)
    print("distance is: {0}".format(distBetween))

    if distBetween <= _max:
        print("SHOULD BE A MATCH: {0}".format(distBetween))
        return True

    return False


def main():

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='mode', help="Mode")
    #create training parser
    trainParser = subparsers.add_parser('train', help="Train a new classifier.")
    #trainParser.add_argument('--groupid', type=str, help="The Group id of the set to be trained")
    #trainParser.add_argument('--personid', type=str, help="The Person id of the person to be trained")
    trainParser.add_argument(
        'imgPath',
        type=str,
        help="the directory where the training images or folders of images are located")

    trainParser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GridSearchSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='LinearSvm')
    

    #create matching parser
    matchParser = subparsers.add_parser(
        'match', help='match 2 faces')
    matchParser.add_argument(
        'faceA',
        type=str,
        help='path to the first face image file')
    matchParser.add_argument(
        'faceB',
        type=str,
        help='path to the second face image file')


    #parse the args! 
    args = parser.parse_args()
    print(args)
    print(args.mode)

    if args.mode == 'match':
        compare_two_faces( os.path.abspath(args.faceA), os.path.abspath(args.faceB) )
    elif args.mode == 'train':
        pre_process_faces(os.path.abspath(args.imgPath))

    


if __name__ == "__main__":
    #sample match
    #should match
    #python face_recognition_example.py match ../data/faces/cam_rogers/0f3ce633-4c88-41e2-aac1-f24044878f8c.jpg ../data/faces/cam_rogers/b8d4f1c3-575b-4c78-9b84-35a0561d1c09.jpg
    #should match
    #python face_recognition_example.py match ../data/faces/cam_rogers/041bedf9-7f02-4722-b744-bacddfa0f8bb.jpg ../data/faces/cam_rogers/b8d4f1c3-575b-4c78-9b84-35a0561d1c09.jpg
    #should not match
    #python face_recognition_example.py match ../data/faces/aaron_bryson/5b85ded0-a6db-4077-be98-d70d9fbf606d.jpg ../data/faces/cam_rogers/b8d4f1c3-575b-4c78-9b84-35a0561d1c09.jpg
    #pre-process folder
    #python face_recognition_example.py train ../data/faces/cam_rogers
    #test on dir of images
    #imgDir = os.path.abspath("../data/dogs")
    #processAll(imgDir)
    main()



"""
#old stuff
imgA = io.imread(faceOne)
    imgB = io.imread(faceTwo)
    imgs = [imgA, imgB]

    #assuming there is one face in each image
    detA = detector(imgA, 1)
    detB = detector(imgB, 1)
    faceDescriptors = []

    # Now process each face we found.
    detections = (detA, detB)
    i=0
    for detection in detections:
        img = imgs[i]
        for k, d in enumerate(detection):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = sp(img, d)
            faceDescriptors.append(facerec.compute_face_descriptor(img, shape))

        i += 1

    if len(faceDescriptors) < 2:
        print("not enough faces found in images")
        exit()
"""