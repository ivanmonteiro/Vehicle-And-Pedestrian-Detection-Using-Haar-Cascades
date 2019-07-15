# -*- coding: utf-8 -*-

import cv2
import imutils
print('Project Topic : Vehicle Classification')
print('Research Internship on Machine learning using Images')
print('By Aditya Yogish Pai and Aditya Baliga B')

video_src = 'fight.mp4'

#cap = cv2.VideoCapture(video_src)
cap = cv2.VideoCapture(0)

pedestrians_cascade = cv2.CascadeClassifier('pedestrian.xml')
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, img = cap.read()

    if (type(img) == type(None)):
        break
    
    img = imutils.resize(img, width=300, height=300)
	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pedestrians = pedestrians_cascade.detectMultiScale(gray,1.1,1, minSize=(20,20))
    #pedestrians = pedestrians_cascade.detectMultiScale(gray,1.1, 1)
    #cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3, minSize=(50,50))

    for(a,b,c,d) in pedestrians:
        cv2.rectangle(img,(a,b),(a+c,b+d),(0,255,255),4)
    
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()
