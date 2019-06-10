import cv2
from sklearn import svm
from sklearn.externals import joblib
import pickle as cPickle
import numpy as np

train_file = "train400.pkl"
labels_file = "labels400.pkl"

print ("Loading Pickle Files")

"""

# with open(train_file, 'rb') as fp:
#     train_data = cPickle.load(fp)

# with open(labels_file, 'rb') as fp:
#     labels_data = cPickle.load(fp)

with open(train_file, 'rb') as fp:
    u = cPickle._Unpickler(fp)
    u.encoding = 'latin1'
    train_data = u.load()
#    train_data = cPickle.load(fp)

with open(labels_file, 'rb') as fp:
    v = cPickle._Unpickler(fp)
    v.encoding = 'latin1'
    labels_data = v.load()
#   labels_data = cPickle.load(fp)

"""
with open(train_file, 'rb') as fp:
  train_data = cPickle.load(fp)

with open(labels_file, 'rb') as fp:
     labels_data = cPickle.load(fp)


clf = svm.LinearSVC()

print ("Training Linear SVM")

clf.fit(train_data, labels_data)

joblib.dump(clf, "model400.pkl")

print ("Model Train and dumped into model400.pkl")


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

# map_dict = {
#     "Adidas" : 1,
#     "Apple" : 2,
#     "BMW" : 3,
#     "Citroen" : 4,
#     "Cocacola" : 5,
#     "DHL" : 6,
#     "Fedex" : 7,
#     "Ferrari" : 8,
#     "Ford" : 9,
#     "Google" : 10,

# }


map_dict = {
    # "Adidas" : 1,
    # "Apple" : 2,
    "BMW" : 1,
    "Citroen" : 2,

    # "Cocacola" : 5,
    # "DHL" : 6,
    # "Fedex" : 7,
    "Ferrari" : 3,
    "Ford" : 4,
    "Mini" :5,
    "Porsche" :6,
    # "Google" : 10,

}

km = joblib.load("cluster400.pkl")

test = "flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
imgpath = "flickr_logos_27_dataset/flickr_logos_27_dataset_images/"
accuracy = []

f = open(test)

for line in f.readlines():
    #imgfile, logoClass = line.split("\t")
    imgfile, logoClass, subset, x1, y1, x2, y2, _  = line.split(" ")
    #logoClass = logoClass.split("\r")[0]

    img = cv2.imread(imgpath + imgfile)
    if logoClass == "Heineken":
        break

    if (int(y2) - int(y1)) < 10:
        continue

    logo = img[int(y1):int(y2), int(x1):int(x2)]

    print (imgfile, logoClass)
    #edge = cv2.Canny(img, 150, 200)

    fd = cv2.xfeatures2d.SIFT_create()

    keypoint, des = fd.detectAndCompute(logo, None)
    if des is not None:
        p = km.predict(des)

        result = clf.predict([np.bincount(p, minlength=400)])
        print (result[0])
        accuracy.append(result[0] == (map_dict[logoClass]))
        #cv2.imshow("image", edge)
        #cv2.waitKey(0)
        
print ("Train Accuracy: " + str(mean(accuracy)*100))
