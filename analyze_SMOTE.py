import json
import cv2
import glob
import numpy as np
from imutils import build_montages
from faces import *

dataset_name = "Caltech"

prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

with open(dataset_name + "_SMOTE.json", "r") as file:
    data = json.load(file)

for name in data:
    faces = []
    print(name + ":")
    image = cv2.imread("new photos/"+ name)
    boxes = detect_faces(net, image, 0.75)
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # extract the face ROI, resize it, and convert it to
        # grayscale
        faceROI = image[startY:endY, startX:endX]
        faceROI = cv2.resize(faceROI, (47, 62))
        faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
    face_set = faceROI
    for item in data[name]:
        print("Name:", item, ", Occurrence:", len(data[name][item]), ", Average Confidence:", sum(data[name][item])/len(data[name][item]))
        if dataset_name == "MUCT":
            path = glob.glob("photos/MUCT/" + item + "**")
        else:
            path = glob.glob("photos/Caltech/" + item + "**")
        filename = path[0]
        image = cv2.imread(filename)
        boxes = detect_faces(net, image, 0.75)
        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # extract the face ROI, resize it, and convert it to
            # grayscale
            faceROI = image[startY:endY, startX:endX]
            faceROI = cv2.resize(faceROI, (47, 62))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
        faces.append(faceROI)
    for face in faces:
        face_set = np.concatenate((face_set, face), axis=1)
    cv2.imshow(name, face_set)
    name = name.split(".")[0]
    cv2.imwrite("results/" + dataset_name + "/" + name + "_SMOTE.jpg", face_set)
    cv2.waitKey(0)
