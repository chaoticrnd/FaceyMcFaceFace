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

#./face_recognition.py ../models/shape_predictor_68_face_landmarks.dat ../models/dlib_face_recognition_resnet_model_v1.dat ../data/faces
"""
You can download a trained facial shape predictor and recognition model from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
"""

"""
TODO:
[] add emotions!!!
[] train data set
[] save and load model
[] test outputs
    http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
    http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/
"""

#Ermagherd!
class Emertional():

    def __init__(self, root, detector, shapePredictor, faceRecModel):
        self.ROOT = root

        self.detector = detector
        self.sp = shapePredictor
        self.facerec = faceRecModel

        self.emotions = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]#, "silly"] #Emotion list
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.clf = svm.SVC(kernel='linear', probability=True, tol=1e-3)#, verbose = True) #Set the classifier as a support vector machines with polynomial kernel
        self.emotionImgDataPath = ""
        self.emoTrainingModelPath = os.path.join(self.ROOT, "models/emoModel.pkl")

        self.totalTrain = 0
        self.numCompleted = 0


    def train(self, dataPath):
        self.emotionImgDataPath = dataPath

        training_data, training_labels = self.make_sets(False)

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

    def feel(self, fileref):
        """
        calculate emotions for input image
        @param: fileref | image path or base64 string
        """
        print('attempting to detect emotion for: ' + fileref)
        #load saved model if can
        if os.path.exists(self.emoTrainingModelPath):
            #load pkl
            self.clf = joblib.load(self.emoTrainingModelPath)
        else:
            print ("error!: missing model, you need to retrain to get model")
            return

        #vectorize new image
        vec = self.proc_emo_img(fileref)
        if vec == None:
            return {"scores":{}, "dominant":"", "confidence":0.0}
        val = self.detect_emo(vec)
        
        return val

    def get_files(self, emotion):
        """
        get 80 20 split of training to testing data
        @param emotion: folder name containing emotion set of .jpgs
        @return  lists of image paths
        """
        #randomly shuffle it and split 80/20
        ep = os.path.join(self.emotionImgDataPath, emotion)
        #print (ep)
        files = glob.glob(ep + "/*")
        #print(files)
        random.shuffle(files)
        return files

    def make_sets(self, split=False):
        """
        @return sets of vectorized faces and labels for training and prediction
        """
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        self.totalTrain = 0
        self.numCompleted = 0
        flz = []
        if split:
            for emotion in self.emotions:
                flz = self.get_files(emotion)
                training = flz[:int(len(flz)*0.8)] #get first 80% of file list
                prediction = flz[-int(len(flz)*0.2):] #get last 20% of file list

                self.totalTrain = (len(training) + len(prediction)) * len(self.emotions)
                #Append data to training and prediction list, and generate labels 0-7
                tset = self.proc_emo_set(emotion, training)
                training_data.append(tset[0])
                training_labels.append(tset[1])
                pset = self.proc_emo_set(emotion, prediction)
                prediction_data.append(pset[0])
                prediction_labels.append(pset[1]) 
              
            return training_data, training_labels, prediction_data, prediction_labels
        else:
            for emotion in self.emotions:
                flz = self.get_files(emotion)
                oneset = flz # flz[:int(len(flz)*0.1)]
                self.totalTrain = len(oneset) * len(self.emotions)
                #Append data to training and prediction list, and generate labels 0-7
                oset = self.proc_emo_set(emotion, oneset)
                training_data = training_data + oset[0]
                training_labels = training_labels + oset[1]
            
            print training_labels
            return training_data, training_labels


    def proc_emo_set(self, emotion, imgset):
        """
        @param: list of image paths
        @return: one list of vectorized landmarks per image,
                one list of labels for each vector
        """
        print("processing: " + emotion)
        set_data = []
        set_labels = []
        for item in imgset:
            #print(item)
            landmarks_vectorised = self.proc_emo_img(item)
            if landmarks_vectorised == None:
                print("landmarks error...")
                pass
            else:
                set_data.append(landmarks_vectorised) #append image array to training data list
                set_labels.append(self.emotions.index(emotion))

            self.numCompleted += 1
            futils.print_progress(self.numCompleted, self.totalTrain)

        return set_data, set_labels


    def proc_emo_img(self, fileref):
    	"""
    	open image from path or base64
    	and return vectorized landmarks
    	@return list
    	"""
    	image = futils.data_to_img(fileref)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
        clahe_image = self.clahe.apply(gray)
        landmarks_vectorised = futils.process_face(self.detector, self.sp, self.facerec, clahe_image, False, True)['landmarkData']['vectorized']
        return landmarks_vectorised

    def run_benchmark(self, i):
    	"""
    	see how accurate the system is
    	"""
        print("Making sets %s" %i) #Make sets by random sampling 80/20%
        training_data, training_labels, prediction_data, prediction_labels = self.make_sets(True)

        #load saved model if can
        if os.path.exists(self.emoTrainingModelPath):
            #load pkl
            self.clf = joblib.load(self.emoTrainingModelPath)
        else:
            #need to train first
            self.train_emo(training_data, training_labels)

        print("getting accuracies %s" %i)
        #Use score() function to get accuracy
        npar_pred = np.array(prediction_data)
        pred_lin = self.clf.score(npar_pred, prediction_labels)
        print "linear: ", pred_lin

        return pred_lin


    def detect_emo(self, prediction_data):
        """
        @param prediction_data: list of vectorized faces
        """

        #check the probablities of each label to this image set
        classRes = self.clf.predict_proba(np.array(prediction_data))
        final = {"scores":{}, "dominant":"", "confidence":0.0}
        for res in classRes:
            i=0
            dom = ""
            hv = 0
            for val in res:
                lbl = self.emotions[i]
                #dat = {}
                #dat[lbl] = val
                #print("prob {0}".format(dat))
                final["scores"][lbl] = val
                if val > hv:
                    hv = val
                    dom = lbl

                i+=1
        final["dominant"] = dom
        final["confidence"] = hv
        #print("prediction {0}".format(classRes))
        return final

    



