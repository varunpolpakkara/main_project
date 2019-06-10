import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.cluster import KMeans
import pickle as cPickle
import sys

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


annotations = "flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt"
imgpath = "flickr_logos_27_dataset/flickr_logos_27_dataset_images/"


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)



f = open(annotations)

ratio = []
diffy = []
diffx = []
features = []
labels = []
all_des = []

count = 0

for line in f.readlines():
    imgfile, logoClass, subset, x1, y1, x2, y2, _  = line.split(" ")
    img = cv2.imread(imgpath + imgfile)

    if logoClass == "Heineken":
        break

    if (int(y2) - int(y1)) < 10:
        continue

    logo = img[int(y1):int(y2), int(x1):int(x2)]

    #ratio.append((int(x2) - int(x1)*1.0)/(int(y2) - int(y1)))
    #diffy.append(int(y2) - int(y1))
    #diffx.append(int(x2) - int(x))

    #resized = cv2.resize(gray, (48, 32), interpolation=cv2.INTER_CUBIC)
    #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #edge = cv2.Canny(logo,150,200)
    #eHist = cv2.equalizeHist(gray)



    #cv2.imshow("logo", logo)

    #cv2.imshow("edge", edge)
    #cv2.imshow("otsu", thresh)
    #cv2.imshow("resized", resized)
    #cv2.waitKey(0)
    #f = resized.ravel()
    fd = cv2.xfeatures2d.SIFT_create()
    keypoint, des = fd.detectAndCompute(logo, None)

    if des is not None:
        count = count + 1
        #print "Extracting Features, Image Count: " + str(count)
        sys.stdout.write("Extracting Features, Image Count: %d / 1644 \r" % count)
        sys.stdout.flush()

        if len(all_des) == 0:
            all_des = des
        else:
    # append the new number to the existing array at this slot
            all_des = np.append(all_des, des, 0)


        #labels.append(map_dict[logoClass])

sys.stdout.write("\n")
print("Done")



with open("des.pkl", 'wb') as fp:
    cPickle.dump(all_des, fp)

centre_dict = {}

print ("Developing Kmeans Cluster Model for clustering Features")

km = KMeans(n_clusters=400, random_state=0)
data = np.array(all_des)
km.fit(data)

joblib.dump(km, "cluster400.pkl")


print ("Model Developed and dumped")




train = []
labels = []

f_again = open(annotations)
count = 0

for line in f_again.readlines():
    imgfile, logoClass, subset, x1, y1, x2, y2, _  = line.split(" ")
    img = cv2.imread(imgpath + imgfile)
    if logoClass == "Heineken":
        break

    if (int(y2) - int(y1)) < 10:
        continue

    logo = img[int(y1):int(y2), int(x1):int(x2)]

    #ratio.append((int(x2) - int(x1)*1.0)/(int(y2) - int(y1)))
    #diffy.append(int(y2) - int(y1))
    #diffx.append(int(x2) - int(x))
    #gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    #resized = cv2.resize(gray, (48, 32), interpolation=cv2.INTER_CUBIC)
    #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #edge = cv2.Canny(logo,150,200)
    #eHist = cv2.equalizeHist(gray)

    fd = cv2.xfeatures2d.SIFT_create()

    keypoint, des = fd.detectAndCompute(logo, None)
    if des is not None:
        count = count + 1
        p = km.predict(des)
        sys.stdout.write("Clustering Features, Image Count: %d /1644 \r" % count)
        sys.stdout.flush()
        train.append(np.bincount(p, minlength=400))
        labels.append(map_dict[logoClass])


sys.stdout.write("\n")

with open("train400.pkl", 'wb') as fp:
    cPickle.dump(train, fp)
with open("labels400.pkl", 'wb') as fp:
    cPickle.dump(labels, fp)

print ("Training set Size:" + str((len(train), len(train[0]))))



print ("Features Extracted and dataset created into train400.pkl and labels400.pkl")
