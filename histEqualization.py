import numpy as np
import cv2
import cv2.cv as cv
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal
airborne = cv2.imread("airborne.jpg",0)

#print airborne

print airborne[12,200]
#cv2.imshow('airborne',airborne)
#cv2.waitKey(0)

histogramFrequencyValues = [0]*256 
frequencyOfIntensity = [0]*256
cumulativeFrequencyOfIntensity = [0]*256

imageList = [0]*(airborne.shape[0]*airborne.shape[1])
def histogramFrequency(imageMatrix): 		# this function calculates frequency of each intensity pixel.
	countPixelIntensity = 0;

	for i in range(imageMatrix.shape[0]):
		for j in range(imageMatrix.shape[1]):
		#	if(imageMatrix[i][j] == intensity):
			frequencyOfIntensity[imageMatrix[i][j]] = frequencyOfIntensity[imageMatrix[i][j]] + 1
			
	return frequencyOfIntensity

def imageMatrixToList(imageMatrix): 		# this function converts the 2D matrix into a list for plotting a graph
	for i in range(imageMatrix.shape[0]):
  	       for j in range(imageMatrix.shape[1]):
			imageList.append(imageMatrix[i][j])
	return imageList

#print imageMatrixToList(airborne)			

histogramFrequencyValues = histogramFrequency(airborne)

#print histogramFrequencyValues

def cummulativeFrequency(histFrequency):
	cumulativeFrequency = [0]*256
	for i in range(256):
		if(i==0):
			cumulativeFrequency[i] =  histFrequency[i]
		else:
			cumulativeFrequency[i] = cumulativeFrequency[i-1] + histFrequency[i]

	return cumulativeFrequency

print "******Cummulative frequency*******"
cumulativeFrequencyOfIntensity =  cummulativeFrequency(histogramFrequencyValues)	

print cumulativeFrequencyOfIntensity

plt.hist(imageMatrixToList(airborne),bins = 256)
plt.title("Image histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
#plt.show()

equilizedImage = np.zeros([airborne.shape[0],airborne.shape[1]])
#equilizedImage = np.matrix(equilizedImageArray)
#print equilizedImage
def histEquilization(img):
	print "cumulativeFrequencyOfIntensity[img[0][0]] ", cumulativeFrequencyOfIntensity[img[0][0]]
	print "img.shape[0] * img.shape[1] ", img.shape[0] * img.shape[1]
	
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			equilizedImage[i][j] = (256*cumulativeFrequencyOfIntensity[img[i][j]])/(img.shape[0] * img.shape[1])

#			print equilizedImage.shape
#			print i,j	
	return equilizedImage

equilizedImage = histEquilization(airborne)
print equilizedImage


cv2.imshow('Airborne Equilized',equilizedImage)
cv2.waitKey(0)

#print "Count of pixels with intensity x:",searchIntensityValues(airborne,129)	

#print len(frequencyOfIntensity)

#def histogramFrequency(inputImage):
#	for intensity in range(255):
#		frequencyOfIntensity[intensity] = searchIntensityValues(inputImage,intensity)
		
#	print frequencyOfIntensity

#histogramFrequency(airborne)

		
