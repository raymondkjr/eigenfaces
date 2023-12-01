# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from faces import load_face_dataset, load_one_face
import numpy as np
import cv2
import json
import os

# set all the arguments for running the analysis
args = {
    "input": "MUCT",
    "confidence": 0.75,
    "num_components": 150,
    "num_iters": 50,
    "test_photo": [f for f in os.listdir("new photos") if f != ".DS_Store"]
}

if args["input"] == "MUCT":
    C = 100
    gamma = 0.001
else:
    C = 10
    gamma = 0.001
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the MUCT faces dataset
print("[INFO] loading dataset...")
(faces, labels) = load_face_dataset("photos/"+args["input"], net,
                                    minConfidence=0.5, minSamples=10)
print("[INFO] {} images in dataset".format(len(faces)))
# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])
# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# load a new photo that is not part of the set
newFaces = []
results = {}
for file in args["test_photo"]:
    newFace = load_one_face("new photos/"+file, net)
    newFaces.append(newFace)
    results[file] = {}
# create PCA
pca = PCA(
    svd_solver="randomized",
    n_components=args["num_components"],
    whiten=True)
# create SVC for classification
model = SVC(kernel="rbf", C=C, gamma=gamma, random_state=42, probability=True)
# loop 20 times

for i in range(args["num_iters"]):
    # construct our training and testing split
    split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
                             stratify=labels)
    (origTrain, origTest, trainX, testX, trainY, testY) = split

    # fit all X to PCA
    trainX = pca.fit_transform(trainX)
    # train a classifier on the eigenfaces representation
    # fit the model using training data
    model.fit(trainX, trainY)
    for idx, newFace in enumerate(newFaces):
        # convert one face using the pca transform
        newFaceX = pca.transform(np.reshape(newFace, (1,-1)))
        # predict the new sample
        newPred = model.predict(newFaceX)
        predName = le.inverse_transform(newPred)[0]

        # determine the class probabilities for the new predicted face
        predProb = model.predict_proba(newFaceX)
        # determine top 10 probabilities and classes to normalize the probabilities
        sortedProbInd = list(np.argsort(predProb)[0])
        sortedProbInd.reverse()
        top10 = predProb[0][sortedProbInd[:20]]
        total = sum(top10)
        maxProbInd = sortedProbInd[0]
        # calculate the normalized probability of the prediction
        normProb = predProb[0][maxProbInd]/total*100
        # add prediction to results dictionary
        if predName in results[args["test_photo"][idx]]:
            results[args["test_photo"][idx]][predName].append(normProb)
        else:
            results[args["test_photo"][idx]][predName] = [normProb]
    # # visualize the predicted face next to the original face
    # predFace = np.dstack([faces[newPred][0]]*3)
    # predFace = imutils.resize(predFace, width=250)
    # actualFace = imutils.resize(np.dstack([newFace]*3), width=250)
    # predictionImages = [predFace, actualFace]
    # predictionMontage = build_montages(predictionImages, (250,250),(2,1))[0]
    # cv2.imshow("Prediction", predictionMontage)
    # cv2.waitKey(0)
    print("[INFO]: {:.2f}% done".format((i+1)/args["num_iters"]*100))

with open(args["input"]+".json", "w") as file:
    json.dump(results, file)