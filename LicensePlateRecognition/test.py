import cv2
import numpy as np
from sklearn.externals import joblib


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


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

map_dict_result = {
    1 : "BMW",
    2 : "Citroen",
    3 : "Ferrari",
    4 : "Ford",
    5 : "Mini",
    6 : "Porsche"
}

def testing():
    clf = joblib.load("./LicensePlateRecognition/model400.pkl")
    km = joblib.load("./LicensePlateRecognition/cluster400.pkl")

    #test = "flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt"
    # imgpath = "flickr_logos_27_dataset/flickr_logos_27_dataset_images/"
    accuracy = []

    #f = open(test)

    # for line in f.readlines():
    #     imgfile, logoClass, _ = line.split(" ")
        
        # imgfile = cv2.imread('l3.jpg')
        # logoClass = "Ford"
        #imgfile, logoClass, subset, x1, y1, x2, y2, _  = line.split(" ")
        #logoClass = logoClass.split("\r")[0]
        # print ("2")
    logoClass = "Ford"
    # imgpath = './flickr_logos_27_dataset/flickr_logos_27_dataset_images/'
    # imgfile = 'l14.jpg'

    # imgpath = './test/'
    # imgfile = 'l8.jpg'
    # img = cv2.imread(imgpath + imgfile)

    # imgpath = './test/'
    imgfile = './static/outputs/logo1.jpg'
    img = cv2.imread(imgfile)

    # if logoClass == "Heineken":
    #     break

        # print (imgfile, logoClass)
        #edge = cv2.Canny(img, 150, 200)

    fd = cv2.xfeatures2d.SIFT_create()

    keypoint, des = fd.detectAndCompute(img, None)
    if des is not None:
        #print ("1")
        p = km.predict(des)

        result = clf.predict([np.bincount(p, minlength=400)])
        #print ('Result', result[0])
        accuracy.append(result[0] == (map_dict[logoClass]))
        #cv2.imshow("image", edge)
        #cv2.waitKey(0)
        
    # print ('Result Final', result[0])
    #print('Result Final',map_dict_result[result[0]])
    # print ("Accuracy: " + str(mean(accuracy)*100))
    return map_dict_result[result[0]]

if __name__ == "__main__":
    # importlib.reload(Extraction)
    # im = readInput()
    # cv2.imshow('File',im)
    testing()
    cv2.waitKey(0)