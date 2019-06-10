from __future__ import division 
from TOOLS import Functions


import math
import cv2
import numpy as np
import Main
import argparse
import matplotlib.pyplot as plt

global imgT

plate=[] 


def process(img):
# this folder is used to save the image
    # temp_folder = 'C:\\Users\\PC\\Desktop\\temp\\'

    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
    # args = vars(ap.parse_args())

    # img = cv2.imread(args["image"])
    # cv2.imshow('original', img)
    # cv2.imwrite(temp_folder + '1 - original.png', img)

    # hsv transform - value = gray image

    # img = cv2.imread(img_location)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # cv2.imshow('HSV',hsv)
    # cv2.imwrite('HSV.jpg',hsv)
    # cv2.imshow('gray', value)
    # cv2.imwrite(temp_folder + '2 - gray.png', value)

    # kernel to use for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # applying topHat/blackHat operations
    topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
    blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)
    #cv2.imshow('topHat', topHat)
    #cv2.imshow('blackHat', blackHat)
    # cv2.imwrite(temp_folder + '3 - topHat.png', topHat)
    # cv2.imwrite(temp_folder + '4 - blackHat.png', blackHat)

    # add and subtract between morphological operations
    add = cv2.add(value, topHat)
    subtract = cv2.subtract(add, blackHat)
    # cv2.imshow('subtract', subtract)
    # cv2.imwrite(temp_folder + '5 - subtract.png', subtract)

    # applying gaussian blur on subtract image
    blur = cv2.GaussianBlur(subtract, (5, 5), 0)
    # cv2.imshow('blur', blur)
    # cv2.imwrite(temp_folder + '6 - blur.png', blur)

    # thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    # cv2.imshow('thresh', thresh)
    # cv2.imwrite(temp_folder + '7 - thresh.png', thresh)

    # check for contours on thresh
    imageContours, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # get height and width
    height, width = thresh.shape

    # create a numpy array with shape given by threshed image value dimensions
    imageContours = np.zeros((height, width, 3), dtype=np.uint8)

    # list and counter of possible chars
    possibleChars = []
    countOfPossibleChars = 0

    # loop to check if any (possible) char is found
    for i in range(0, len(contours)):

        # draw contours based on actual found contours of thresh image
        cv2.drawContours(imageContours, contours, i, (255, 255, 255))

        # retrieve a possible char by the result ifChar class give us
        possibleChar = Functions.ifChar(contours[i])

        # by computing some values (area, width, height, aspect ratio) possibleChars list is being populated
        if Functions.checkIfChar(possibleChar) is True:
            countOfPossibleChars = countOfPossibleChars + 1
            possibleChars.append(possibleChar)

    # cv2.imshow("contours", imageContours)
    # cv2.imwrite(temp_folder + '8 - imageContours.png', imageContours)

    imageContours = np.zeros((height, width, 3), np.uint8)

    ctrs = []

    # populating ctrs list with each char of possibleChars
    for char in possibleChars:
        ctrs.append(char.contour)

    # using values from ctrs to draw new contours
    cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))
    # cv2.imshow("contoursPossibleChars", imageContours)
    # cv2.imwrite(temp_folder + '9 - contoursPossibleChars.png', imageContours)

    plates_list = []
    listOfListsOfMatchingChars = []

    for possibleC in possibleChars:

        # the purpose of this function is, given a possible char and a big list of possible chars,
        # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
        def matchingChars(possibleC, possibleChars):
            listOfMatchingChars = []

            # if the char we attempting to find matches for is the exact same char as the char in the big list we are currently checking
            # then we should not include it in the list of matches b/c that would end up double including the current char
            # so do not add to list of matches and jump back to top of for loop
            for possibleMatchingChar in possibleChars:
                if possibleMatchingChar == possibleC:
                    continue

                # compute stuff to see if chars are a match
                distanceBetweenChars = Functions.distanceBetweenChars(possibleC, possibleMatchingChar)

                angleBetweenChars = Functions.angleBetweenChars(possibleC, possibleMatchingChar)

                changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
                    possibleC.boundingRectArea)

                changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
                    possibleC.boundingRectWidth)

                changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
                    possibleC.boundingRectHeight)

                # check if chars match
                if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
                        angleBetweenChars < 12.0 and \
                        changeInArea < 0.5 and \
                        changeInWidth < 0.8 and \
                        changeInHeight < 0.2:
                    listOfMatchingChars.append(possibleMatchingChar)

            return listOfMatchingChars


        # here we are re-arranging the one big list of chars into a list of lists of matching chars
        # the chars that are not found to be in a group of matches do not need to be considered further
        listOfMatchingChars = matchingChars(possibleC, possibleChars)

        listOfMatchingChars.append(possibleC)

        # if current possible list of matching chars is not long enough to constitute a possible plate
        # jump back to the top of the for loop and try again with next char
        if len(listOfMatchingChars) < 3:
            continue

        # here the current list passed test as a "group" or "cluster" of matching chars
        listOfListsOfMatchingChars.append(listOfMatchingChars)

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

        recursiveListOfListsOfMatchingChars = []

        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

        break

    imageContours = np.zeros((height, width, 3), np.uint8)

    for listOfMatchingChars in listOfListsOfMatchingChars:
        contoursColor = (255, 0, 255)

        contours = []

        for matchingChar in listOfMatchingChars:
            contours.append(matchingChar.contour)

        cv2.drawContours(imageContours, contours, -1, contoursColor)

    # cv2.imshow("finalContours", imageContours)
    # cv2.imwrite(temp_folder + '10 - finalContours.png', imageContours)

    for listOfMatchingChars in listOfListsOfMatchingChars:
        possiblePlate = Functions.PossiblePlate()

        # sort chars from left to right based on x position
        listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

        # calculate the center point of the plate
        plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[len(listOfMatchingChars) - 1].centerX) / 2.0
        plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[len(listOfMatchingChars) - 1].centerY) / 2.0

        plateCenter = plateCenterX, plateCenterY

        # calculate plate width and height
        plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
            len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

        totalOfCharHeights = 0

        for matchingChar in listOfMatchingChars:
            totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

        averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

        plateHeight = int(averageCharHeight * 1.5)

        # calculate correction angle of plate region
        opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

        hypotenuse = Functions.distanceBetweenChars(listOfMatchingChars[0],
                                                    listOfMatchingChars[len(listOfMatchingChars) - 1])
        correctionAngleInRad = math.asin(opposite / hypotenuse)
        correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

        # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
        possiblePlate.rrLocationOfPlateInScene = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

        # get the rotation matrix for our calculated correction angle
        rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

        height, width, numChannels = img.shape

        # rotate the entire image
        imgRotated = cv2.warpAffine(img, rotationMatrix, (width, height))

        # crop the image/plate detected
        imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))

        # copy the cropped plate image into the applicable member variable of the possible plate
        possiblePlate.Plate = imgCropped

        # cv2.imshow('Plate',imgCropped)

        cv2.imwrite('./static/outputs/NumberPlate.jpg',imgCropped)
        imgT = imgCropped
        # populate plates_list with the detected plate
        if possiblePlate.Plate is not None:
            plates_list.append(possiblePlate)

        # draw a ROI on the original image
        for i in range(0, len(plates_list)):
            # finds the four vertices of a rotated rect - it is useful to draw the rectangle.
            p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)

            # roi rectangle colour
            rectColour = (0, 255, 0)

            cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
            cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
            cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
            cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

            cv2.line(img, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
            cv2.line(img, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
            cv2.line(img, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
            cv2.line(img, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)
            
            imageContours = cv2.cvtColor(imageContours, cv2.COLOR_BGR2GRAY)
            sobel = cv2.Sobel(imageContours, cv2.CV_8U, 1, 0, ksize = 3)
            
            element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
            dilation = cv2.dilate(sobel, element2, iterations = 1)
            erosion = cv2.erode(dilation, element1, iterations = 1)
            dilation2 = cv2.dilate(erosion, element2,iterations = 10)

            #cv2.imshow("detected", dilation2)
            # cv2.imwrite(temp_folder + '11 - detected.png', imageContours)

            #cv2.imshow("detectedOriginal", img)
            # cv2.imwrite(temp_folder + '12 - detectedOriginal.png', img)

            # cv2.imshow("plate", plates_list[i].Plate)
            # cv2.imwrite(temp_folder + '13 - plate.png', plates_list[i].Plate)
            return dilation2

def plateDetect(img,img2,imgk):
    
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        
        x,y,w,h=cv2.boundingRect(con)    
        area=w*h
        ratio=w/h
        print(area,ratio)
        
        if ratio>1 and ratio<5 and area>=2000 and area<=80000:
            print ('ya')
            logo_y1=max(0,int(y-h*3.0))
            logo_y2=y
            logo_x1=x
            logo_x2=x+w
            img_logo=imgk.copy()
            logo=img_logo[logo_y1:logo_y2,logo_x1:logo_x2]
            cv2.imwrite('./static/outputs/logo1.jpg',logo)
            # cv2.imwrite('./static/outputs/SIFT/flickr_logos_27_dataset/flickr_logos_27_dataset_images/logo.jpg',logo)
            cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.rectangle(img2,(logo_x1,logo_y1),(logo_x2,logo_y2),(0,255,0),2)
            number_plate1 = img_logo[y:y+h, x:x+w]
            global plate
            plate=[x,y,w,h]

            
            width = 200
            height = 100
            dim = (width,height)

            number_plate1 = cv2.resize(number_plate1,dim,interpolation = cv2.INTER_AREA)
            print('Size',number_plate1.shape)
            #cv2.imshow('plateDetect', number_plate1)
            # cv2.imwrite('./number_plate1.jpg',number_plate1)
            
            
            new_image = number_plate1
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
            new_image = np.invert(new_image)
            kernel = np.ones((1, 1), np.uint8)
            new_image = cv2.dilate(new_image, kernel, iterations=1)
            new_image = cv2.erode(new_image, kernel, iterations=1) 
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            new_image = cv2.filter2D(new_image,-1,kernel)
            #new_image = cv2.Canny(new_image, 170, 200)
               
            # cv2.imshow("Final Plate",new_image)
            # cv2.imwrite('./static/outputs/editPlate.jpg',new_image)
           
            # cv2.imshow('numberplate', logo)
            return logo
            

def logoDetect(img,imgo,imgk):
    imglogo=imgo.copy()
    #noplate=img.copy()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(2*img.shape[1],2*img.shape[0]),interpolation=cv2.INTER_CUBIC)
    #img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,-3)
    ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #img=cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 9)
    img=cv2.Canny(img,100,200)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img, element2,iterations = 1)
    img = cv2.erode(img, element1, iterations = 3)
    img = cv2.dilate(img, element2,iterations = 5)

    # cv2.imshow('logoSpec',img)
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    tema=0
    result=[]
    for con in contours:
        x,y,w,h=cv2.boundingRect(con)
        area=w*h
        ratio=max(w/h,h/w)
        # print(area,ratio)
        if area>2000 and area<20000 and ratio<4:
            if area>tema:
                tema=area
                result=[x,y,w,h]
                ratio2=ratio

    logo2_X=[int(result[0]/2+plate[0]-3),int(result[0]/2+plate[0]+result[2]/2+3)]
    logo2_Y=[int(result[1]/2+max(0,plate[1]-plate[3]*3.0)-3),int(result[1]/2+max(0,plate[1]-plate[3]*3.0)+result[3]/2)+3]
    
    cv2.rectangle(img,(result[0],result[1]),(result[0]+result[2],result[1]+result[3]),(255,0,0),2)
    
    cv2.rectangle(imgo,(logo2_X[0],logo2_Y[0]),(logo2_X[1],logo2_Y[1]),(0,0,255),2)
    print (tema,ratio2,result)
    img_plate = im2.copy()
    #print result[0], result[1], result[2], result[3]
    logo2=imglogo[logo2_Y[0]:logo2_Y[1],logo2_X[0]:logo2_X[1]]
    # cv2.imwrite('./static/outputs/logo2.jpg',logo2)

    return img

def trial():
    return imgT

def init(img_filename):
    img = cv2.imread(img_filename)
    imgk = img
    
    plateImg = process(img)
    # cv2.imshow('img',imgk)
    # cv2.imshow('plateimg',plateImg)
    # cv2.imshow('img',img)
    logo = plateDetect(plateImg,img,imgk)
    # cv2.imshow('logo',logo)
    # cv2.imshow('try',img)
    # imgT = trial()
    logo2=logoDetect(logo,img,imgk)
    # cv2.namedWindow('plate',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('plate',600,400)
    # cv2.imshow('Logo',logo)
    # cv2.imshow('plate',img)
    # cv2.imwrite('./ouputs/Extract.jpg',img)
    # cv2.imshow('logo',logo)
    # cv2.waitKey(0)


if __name__ == '__main__':
    # img=cv2.imread('./test/2.jpg')
    img = Main.readInput()
    imgk = img
    
    plateImg = process(img)
    cv2.imshow('img',imgk)
    # cv2.imshow('plateimg',plateImg)
    # cv2.imshow('img',img)
    logo = plateDetect(plateImg,img,imgk)
    cv2.imshow('logo',logo)
    # cv2.imshow('try',img)
    # imgT = trial()
    logo2=logoDetect(logo,img,imgk)
    cv2.namedWindow('plate',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('plate',600,400)
    # cv2.imshow('Logo',logo)
    # cv2.imshow('plate',img)
    # cv2.imwrite('./static/outputs/Extract.jpg',img)
    # cv2.imshow('logo',logo)
    cv2.waitKey(0)
