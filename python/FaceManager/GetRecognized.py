#!/usr/bin/python

"""
Emotion training and recognition
@Author: Matt Murray
"""
import sys, os, glob, json, argparse, pickle, math, random, itertools
import dlib
import cv2
from skimage import io
from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib
from scipy.spatial import distance as dist
import numpy as np
import futils

#Ermagherd
class Rercognerzer():

    def __init__(self, root, detector, shapePredictor, faceRecModel):
        self.ROOT = root

        self.detector = detector
        self.sp = shapePredictor
        self.facerec = faceRecModel

        #progress stuffs
        self.totalTrain = 0
        self.numCompleted = 0


    def make_sets(self, rawData=None):
        """
        make the data into separate label and data lists
        @return data list, lable list
        """
        #example  would be dump of descriptors paired
        #with uuid from db
        if rawData is None:
            rawData = {
                "uuid":[[0.0,0.0,0.0],[0.0,0.0,0.0]]
            }
        data = []
        labels = []
        for k, v in rawData.iteritems():
            data.append(v)
            labels.append(k)

        return data, labels

    def train(self, bulkData):

        training_data, training_labels = self.make_sets(bulkData)

        print("training data:", training_data)
        #print("training labels: ", training_labels)

        #Turn the training set into a numpy array for the classifier
        npar_train = np.array(training_data)
        npar_trainlabs = np.array(training_labels)
        print("training SVM linear")
        #train SVM
        self.clf.fit(npar_train, training_labels)
        #save training as model
        #ref http://scikit-learn.org/stable/modules/model_persistence.html
        joblib.dump(self.clf, self.emoTrainingModelPath)
        print("model saved to: {0}".format(self.emoTrainingModelPath) )
        print("Emotional Trainging Complete")
        #s = pickle.dump(self.emoTrainingModelPath ,self.clf)
        #load
        # self.clf = joblib.load(self.emoTrainingModelPath)
        # self.clf = pickle.load(self.emoTrainingModelPath)

        #accur_lin = []
        #for i in range(0,1):
        #    #test traing and prediction
        #    accur_lin.append(self.run_benchmark(i))

        #print("Mean value lin svm: %.3f" %np.mean(accur_lin)) #Get mean accuracy of the 10 runs

    def identify(self, f):
        """
        Given a new image, generate descriptor and compare to all others?
        or run through classifier
        @param: f | path or base64 image string
        """
        pass
        #compare new to all


    def pre_process_faces(self, facesPath):
        """
        @param: face images dir
        process all images and save data for each one
        """
        #iterate over all images in path and save the ladmarks detected
        allImages = glob.glob(os.path.join(facesPath, "*.jpg"))
        #print(allImages)
        # Now process all the images
        for f in allImages:
            self.process_face(f, True)

    def process_face(self, f):
        """
        @param: face image path
        load a face, get landmarks and features, save to disk
        @return: face descriptors  = 128D vector
        """
        print("Processing file: {}".format(f))

        img = futils.data_to_img(f) #io.imread(f)
        calcDescriptor = True
        resultData = futils.process_face(self.detector, self.sp, self.facerec, img, calcDescriptor)

        #print(face_descriptor)
        #not saving to HD anymore, should be in DB
        #faceDataPath = f.replace('.jpg', '.json')
        #print("saving results to: {0}".format(faceDataPath))
        
        #self.save_features(faceDataPath, resultData)

        return resultData


    def save_features(self, pth, data):
        """
        @param: path to file
        @param: data of face, bounds, descriptors, and landmarks
        """
        with open(pth, 'w') as fdp:
            fdp.write(json.dumps(data))

    def load_features(self, pth):
        """
        @param: file path of data
        @return: previously calculated descriptors
        """
        d = None
        with open(pth, 'r') as f:
            d = json.loads(f.read())

        return d['faceDescriptor']

    def compare(self, faceOne, faceTwo):
        """
        given 2 image paths, load them, find landmarks then compare euclidean
        distance to see if they are similar
        """
        aData = None
        bData = None

        #assume the descriptors have already
        #been calculated for the images but check to be safe

        if type(faceOne) is list:
            #data is in form of a descriptor list
            aData = faceOne
            bData = faceTwo
        else:
            print("this is a file: " + faceOne)
            #don't load form local anymore, either new process or from db
            #dataPathA = faceOne.replace('.jpg', '.json')
            #dataPathB = faceTwo.replace('.jpg', '.json')
            
            #see if the files are there
            #if not os.path.isfile(dataPathA):
            aData = self.process_face(faceOne)['descriptor']
            #else:
            #    aData = self.load_features(dataPathA)

            #if not os.path.isfile(dataPathB):
            bData = self.process_face(faceTwo)['descriptor']
            #else:
            #    bData = self.load_features(dataPathB)


        #compare the resulting descriptors
        return self.compare_descriptors(aData, bData)


    def compare_descriptors(self, distA, distB, _max=0.59):
        tf = False
        distBetween = dist.euclidean(distA, distB)
        print("distance is: {0}".format(distBetween))

        if distBetween <= _max:
            print("SHOULD BE A MATCH: {0}".format(distBetween))
            tf = True

        return {"match":tf, "distance":distBetween}



    



