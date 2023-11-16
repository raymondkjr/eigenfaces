from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from faces import load_face_dataset, load_one_face
import numpy as np
import cv2

# Use sklearn to determine the best SVM model hyperparameters (C, gamma)
args = {
    "input": "Caltech",
    "confidence": 0.5,
    "num_components": 150
}

prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)

(faces, labels) = load_face_dataset("photos/"+args["input"], net,
                                    minConfidence=0.5, minSamples=10)
pcaFaces = np.array([f.flatten() for f in faces])
le = LabelEncoder()
labels = le.fit_transform(labels)

pca = PCA(
    svd_solver="randomized",
    n_components=args["num_components"],
    whiten=True)

trainX = pca.fit_transform(pcaFaces)
trainY = labels
# create SVC for classification

args = {
    "param_grid": {"C": np.logspace(0,8,9),
                   "gamma": np.logspace(-5, 5, 11)},
    "cv": KFold(n_splits=10)
}

model = SVC(kernel="rbf")

validator = GridSearchCV(estimator=model, **args)
validator.fit(trainX, trainY)

print(validator.best_params_)