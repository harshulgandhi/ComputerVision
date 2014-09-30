import numpy as np
import cv2
import cv2.cv as cv
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal
import math
from time import time

labPhoto = cv2.imread("labPhoto.JPG",0)
labPhotoColored = cv2.imread("labPhoto.JPG",1)
chessBoard = cv2.imread("Chess_Board.jpg",0)
chessBoardColored = cv2.imread("Chess_Board.jpg",1)

lambdas = []
x = []
y = []

minLambda = 0
lambdaWithPoints = [[0 for a in range(3)] for b in range(((labPhoto.shape[0]*labPhoto.shape[1])/338)-81)]


def goodFeatureDetector(img):					# function to parse entire image window
	for i in range (0,img.shape[0]-13,13):			# by window. Window is of size 3x3, hence
		for j in range (0,img.shape[1]-26,26):		# step size of 3.
			minLambda = calculateLambdaForWindow(img,i,j)
			lambdas.append(minLambda)
			x.append(i+7)
			y.append(j+7)		
	
			
	
	for i in range(0,len(lambdas)):				#combining x,y coordinates and corresponding lambda
		lambdaWithPoints[i][0] = lambdas[i]		#values in one list.
		lambdaWithPoints[i][1] = x[i]
		lambdaWithPoints[i][2] = y[i]
	print lambdaWithPoints
	bubbleSort(lambdaWithPoints)
	#for i in range(0,len(lambdaWithPoints)):
	 #       if(lambdaWithPoints[i][0] == None):
	  #     	        lambdaWithPoints[i][0] = lambdaWithPoints[0][0]

	formSqaure(bubbleSort(lambdaWithPoints),labPhotoColored)
#	print lambdaWithPoints
#	print len(lambdaWithPoints)
#	print "*****SORTED******"
	
def bubbleSort(lambdaWithPoints):
	i = 0
	while (i<len(lambdaWithPoints)):
		for j in range(len(lambdaWithPoints)):
			if(i<len(lambdaWithPoints) and lambdaWithPoints[i][0]>lambdaWithPoints[j][0]):
				temp1 = lambdaWithPoints[i][0]
				temp2 = lambdaWithPoints[i][1]
				temp3 = lambdaWithPoints[i][2]
				lambdaWithPoints[i][0] = lambdaWithPoints[j][0]
				lambdaWithPoints[i][1] = lambdaWithPoints[j][1]
				lambdaWithPoints[i][2] = lambdaWithPoints[j][2]
				lambdaWithPoints[j][0] = temp1
				lambdaWithPoints[j][1] = temp2
				lambdaWithPoints[j][2] = temp3

			
		i += 1
	return lambdaWithPoints
							

def formSqaure(lambdaWithPoints,image):
	for i in range(250):
#		cv2.circle(image,(lambdaWithPoints[i][1]-7,lambdaWithPoints[i][2]-7),8,(0,0,255,0),2,4,0)
		cv2.rectangle(image,(lambdaWithPoints[i][2]-7,lambdaWithPoints[i][1]-7),(lambdaWithPoints[i][2]+7,lambdaWithPoints[i][1]+7),(0,0,255,0),2)
#	cv2.imwrite('ChessBoardColored.jpg',image)
	cv2.imwrite('Cornered photo.jpg',image)


def calculateLambdaForWindow(img,i,j):

	Ix = 0
	Iy = 0
	sum_Ix = 0
	sum_Iy = 0
	sum_IxIy = 0
#	print "img shape [0] ",img.shape[0]
#	print "img shape [1] ",img.shape[1]
	for m in range (i,i+13):
		for n in range (j,j+13):
			if((img[m,n]<img[m,n+1])):		#Calculation for Ix. If condition because
				Ix = (img[m,n]-img[m,n+1])	#Python is unable to handle negative values
				Ix = (256-Ix)			#and keeps on giving output with repect to value
				Ix = -Ix			#256.
				sum_Ix += (Ix)**2		##Summation of Ix
			elif((img[m,n]>=img[m,n+1])):
				Ix = (img[m,n]-img[m,n+1])
				sum_Ix += (Ix)**2
				
			if((img[m,n]<img[m+1,n])):		#Calculation for Iy
                                Iy = (img[m,n]-img[m+1,n])
                                Iy = (256-Iy)
                                Iy = -Iy
                                sum_Iy += (Iy)**2		## Summation of Iy
                        elif((img[m,n]>=img[m+1,n])):
                                Iy = (img[m,n]-img[m+1,n])
                                sum_Iy += (Iy)**2

			sum_IxIy += Ix*Iy

#			print "Ix =",Ix
#			print "Iy =",Iy
#			print "Ix.Iy =",Ix*Iy

	return calculateMinLambda(sum_Ix/9,sum_Iy/9,(sum_IxIy**2)/9)		#the 1/N factor - size of the window 
											#under consideration
							

def calculateMinLambda(sum_Ix_square,sum_Iy_square,sum_IxIy_square):
	ss_Ix = sum_Ix_square					#Shortening the variabl names so that
	ss_Iy = sum_Iy_square					#equation fits into one line
	ss_IxIy = sum_IxIy_square
	#print "sum_Ix_square  ",sum_Ix_square
	#print "sum_Iy_square  ",sum_Iy_square
	#print "sum_IxIy_square  ",sum_IxIy_square
	b = -(ss_Ix + ss_Iy)
	b_square = (ss_Ix + ss_Iy)**2
	fourac = 4*(ss_Ix*ss_Iy - ss_IxIy)
	lambda_1 = (-b + (b**2 - fourac)**0.5)/2
	lambda_2 = (-b - (b**2 - fourac)**0.5)/2


	if(lambda_1>lambda_2):
		return round(lambda_2,2)
	elif(lambda_1<=lambda_2):
		return round(lambda_1,2)

	
#calculateLambdaForWindow(labPhoto,0,14)
init_time = time()
chessBoard.dtype = np.int8
#labPhoto.dtype = np.int8
labPhotoEqualized = cv2.equalizeHist(labPhoto)
goodFeatureDetector(labPhoto)
print "*****************RUN DURATION********************"
print time() - init_time


