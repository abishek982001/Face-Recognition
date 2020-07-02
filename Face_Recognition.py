import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime

class FaceRecognition:
    images=[]
    classNames=[]
    encodeList=[]
    nameList=[]

    def __init__(self, path, fnames):
        """Parameterised Constructor for initialising path and files"""
        self.path = path
        self.fnames = fnames
        print(self.fnames)
    
    def fileNames(self):
        """Function to read the image and get the filenames without extension.
           This name is assumed as the users name and is used during face recognition"""
        for name in self.fnames:
            curImg = cv2.imread('{}/{}'.format(self.path, name))
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(name)[0])
        print(self.classNames)
        
    def findEncodings(self,images):
        """This function is used to calculate the encodings of a image using face_recognition library"""
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.encodeList.append(encode)
        

    def initiateRecognition(self):
        """This function starts the web cam and gets the encodings from the findEncodings function and uses it to
           calculate the distance between the encodings already found and the current encoding to find a match."""
        encodeList = self.findEncodings(self.images)  
        print("Encoding Complete")
        cap = cv2.VideoCapture(0)
        while True:
            success, img = cap.read()
            imgS = cv2.resize(img, (0,0), None, 0.25, 0.25) # Frame size=1/4th of original size
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
            for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeList, encodeFace) # returns True if match is found
                faceDis = face_recognition.face_distance(self.encodeList, encodeFace) # Returns the distance between encodings
                print(faceDis)
                matchIndex = np.argmin(faceDis)  # Returns the index having minimum distance between the known encodings

                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    print(name)
                    y1,x2,y2,x1 = faceLoc
                    y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4  # To revert the earlier resize in line 44, so that the drawn box fits the face in the current frame size
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                    cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                    self.enter_name(name)

            cv2.imshow('Web Cam', img)
            cv2.waitKey(1)

    def enter_name(self,name):
        """Function to put a entry of the identified person with time in a csv file"""
        with open('Entry.csv', 'r+') as f:
            myDataList = f.readlines()
            for line in myDataList:
                entry = line.split(',')
                self.nameList.append(entry[0])
            if name not in self.nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines('\n{},{}'.format(name, dtString))
