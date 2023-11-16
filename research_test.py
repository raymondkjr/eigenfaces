# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from faces import load_face_dataset, load_one_face
from imutils import build_montages
import numpy as np
import imutils
import time
import cv2

# set all the arguments for running the analysis
args = {
    "input": "photos/MUCT",
    "confidence": 0.5,
    "num_components": 150,
    "visualize": -1,
    "test_photo": "ray.jpg"
}
# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = "deploy.prototxt"
weightsPath = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the MUCT faces dataset
print("[INFO] loading dataset...")
(faces, labels) = load_face_dataset(args["input"], net,
                                    minConfidence=0.5, minSamples=10)
print("[INFO] {} images in dataset".format(len(faces)))
# flatten all 2D faces into a 1D list of pixel intensities
pcaFaces = np.array([f.flatten() for f in faces])
# encode the string labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)
# construct our training and testing split
split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
                         stratify=labels)
(origTrain, origTest, trainX, testX, trainY, testY) = split
# compute the PCA (eigenfaces) representation of the data, then
# project the training data onto the eigenfaces subspace
print("[INFO] creating eigenfaces...")
pca = PCA(
    svd_solver="randomized",
    n_components=args["num_components"],
    whiten=True)
start = time.time()
trainX = pca.fit_transform(trainX)
end = time.time()
print("[INFO] computing eigenfaces took {:.4f} seconds".format(
    end - start))
# check to see if the PCA components should be visualized
if args["visualize"] > 0:
    # initialize the list of images in the montage
    images = []
    # loop over the first 16 individual components
    for (i, component) in enumerate(pca.components_[:16]):
        # reshape the component to a 2D matrix, then convert the data
        # type to an unsigned 8-bit integer so it can be displayed
        # with OpenCV
        component = component.reshape((62, 47))
        component = rescale_intensity(component, out_range=(0, 255))
        component = np.dstack([component.astype("uint8")] * 3)
        images.append(component)
    # construct the montage for the images
    montage = build_montages(images, (47, 62), (4, 4))[0]
    # show the mean and principal component visualizations
    # show the mean image
    mean = pca.mean_.reshape((62, 47))
    mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
    cv2.imshow("Mean", mean)
    cv2.imshow("Components", montage)
    cv2.waitKey(0)
# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, probability=True)
model.fit(trainX, trainY)
# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print(classification_report(testY, predictions,
                            target_names=le.classes_))

# load a new photo that is not part of the set
newFace = load_one_face("new photos/"+args["test_photo"], net)

# convert one face using the pca transform
newFaceX = pca.transform(np.reshape(newFace, (1,-1)))

# predict the new sample
newPred = model.predict(newFaceX)
predName = le.inverse_transform(newPred)[0]

# determine the class probabilities for the new predicted face
predProb = model.predict_proba(newFaceX)
maxProbInd = np.argmax(predProb)
print("[PRED] Predicted Face:", predName)
print("[PRED] Prediction Confidence:{:.2f}%".format(predProb[0][maxProbInd]*100))

# visualize the predicted face next to the original face
predFace = np.dstack([faces[newPred][0]]*3)
predFace = imutils.resize(predFace, width=250)
actualFace = imutils.resize(np.dstack([newFace]*3), width=250)
predictionImages = [predFace, actualFace]
predictionMontage = build_montages(predictionImages, (250,250),(2,1))[0]
cv2.imshow("Prediction", predictionMontage)
cv2.waitKey(0)
