"""
Created on 2018 7.2
@author: ShawnFu
"""

import cv2
import os
import numpy
import time
import readPBModel as PB
import config

font = cv2.FONT_HERSHEY_SIMPLEX
size = 0.5
fx = 10
fy = 355
fh = 18
x0 = 300
y0 = 100
width = IMAGE_WIDTH
height = IMAGE_HEIGHT
numofsamples = num_of_samples
counter = 0
gesturename = ''
path = ''
binaryMode = False
saveImg = False
predi = False

def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (0, 255, 0))
	roi = frame[y0: y0+height,x0: x0+width]
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 2)
	th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
	if saveImg == True and binaryMode == True:
		saveROI(res)
	elif saveImg == True and binaryMode == False:
		saveROI(roi)
	return res
    
def saveROI(img):
	global path, counter, gesturename, saveImg
	if counter > numofsamples:
		saveImg = False
		gesturename = ''
		counter = 0
		return
	counter += 1
	name = gesturename + str(counter) 
	print("Saving img: ", name)
	cv2.imwrite(path+name+'.png', img) 
	time.sleep(1)

	counter += 1
	name = gesturename + str(counter) 
	print("Saving img: ", name)
	h_flip = cv2.flip(img,1)
	cv2.imwrite(path+name+'.png', h_flip) 
	time.sleep(0.05)
'''
	counter += 1
	name = gesturename + str(counter) 
	print("Saving img: ", name)
	v_flip = cv2.flip(img,0)
	cv2.imwrite(path+name+'.png', v_flip) 
	time.sleep(0.05)
'''
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	frame = cv2.flip(frame, 2)
	#cv2.imshow('test',frame)
	roi = binaryMask(frame,x0,y0,width,height)
 	
	cv2.putText(frame,"option:",(fx,fy), font,size,(0,255,0))
	cv2.putText(frame,"b-Binary/r-RGB",(fx,fy+fh), font,size,(0,255,0))
	cv2.putText(frame,"p-prediction",(fx,fy+2*fh), font,size,(0,255,0))
	cv2.putText(frame,"s-new gesture(twice)",(fx,fy+3*fh), font,size,(0,255,0))
	cv2.putText(frame,"q-quit",(fx,fy+4*fh), font,size,(0,255,0))
 	
	key = cv2.waitKey(1) & 0xFF
	if key == ord('b'):
		binaryMode = True
		print("Binary Active")
	elif key == ord('r'):
		binaryMode = False
 		
	if key == ord('i'):
		y0 = y0 - 5
	elif key == ord('k'):
		y0 = y0 + 5 
	elif key == ord('j'):
		x0 = x0 - 5 
	elif key == ord('l'):
		x0 = x0 + 5 
 		
	if key == ord('p'):
		predi = True
		if predi:
			rob = frame[y0: y0+height,x0: x0+width]
			PB.predict(rob)
		predi = False

	if key == ord('q'):
		break
 	
	if key == ord('s'):
		if gesturename != '':
			saveImg = True
			#saveROI(frame)
		else:
			print("please enter N first")
			saveImg = False
	elif key == ord('n'):
		gesturename = input("gesture name(folder)")
		os.makedirs(gesturename)
		path = "./"+gesturename+"/"
 	
	cv2.imshow('frame',frame)
	if(binaryMode):
		cv2.imshow('ROI',roi)
	else:
		cv2.imshow('ROI',frame[y0:y0+height,x0:x0+width])
 
cap.release()
cv2.destroyAllWindows()
 
 

