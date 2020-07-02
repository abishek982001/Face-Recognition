import Face_Recognition
import os

path=''
try:
    path = input("Enter the path of the images: ")
    fnames = os.listdir(path)
except:
    print("Invalid Path")
    exit(0)

faceRecognition = Face_Recognition.FaceRecognition(path, fnames)
faceRecognition.fileNames()
faceRecognition.initiateRecognition()
