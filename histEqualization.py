import numpy as np
import cv2
import cv2.cv as cv
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal
airborne = cv2.imread("airborne.jpg",0)

#print airborne

histogramFrequencyValues = [0]*256 
frequencyOfIntensity = [0]*256
cumulativeFrequencyOfIntensity = [0]*256

imageList = [0]*(airborne.shape[0]*airborne.shape[1])
def histogramFrequency(imageMatrix): 		# this function calculates frequency of each intensity pixel.
	countPixelIntensity = 0;

	for i in range(imageMatrix.shape[0]):
		for j in range(imageMatrix.shape[1]):
			frequencyOfIntensity[imageMatrix[i][j]] = frequencyOfIntensity[imageMatrix[i][j]] + 1
			
	return frequencyOfIntensity

def imageMatrixToList(imageMatrix): 		# this function converts the 2D matrix into a list for plotting a graph
	for i in range(imageMatrix.shape[0]):
  	       for j in range(imageMatrix.shape[1]):
			imageList.append(imageMatrix[i][j])
	return imageList

#print imageMatrixToList(airborne)			

histogramFrequencyValues = histogramFrequency(airborne)
#print "histogramFrequencyValues[192] ",histogramFrequencyValues[192]
#print histogramFrequencyValues

def cummulativeFrequency(histFrequency):	#calculates the cumulative frequency of each intensity value and returns a list
	cumulativeFrequency = [0]*256		#with cumulative frequency populated corresponding to each index representing
	for i in range(256):			#intensity value
		if(i==0):
			cumulativeFrequency[i] =  histFrequency[i]
		else:
			cumulativeFrequency[i] = cumulativeFrequency[i-1] + histFrequency[i]

	return cumulativeFrequency

#print "******Cummulative frequency*******"

cumulativeFrequencyOfIntensity =  cummulativeFrequency(histogramFrequencyValues)	

#print cumulativeFrequencyOfIntensity

plt.hist(imageMatrixToList(airborne),bins = 256) # to plot histogram
plt.title("Image histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

#equilizedImage = np.zeros([airborne.shape[0],airborne.shape[1]])
equilizedImage = [[0 for x in xrange(960)] for x in xrange(720)]

#equilizedImage = np.matrix(equilizedImageArray)
#print equilizedImage
def histEquilization(img):
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			equilizedImage[i][j] = (255*cumulativeFrequencyOfIntensity[img[i][j]])/(img.shape[0] * img.shape[1])
			
	return equilizedImage

equilizedImage = histEquilization(airborne)
finalEqualizedImage = np.array(equilizedImage) # converting equalized 2D array into numpy array so that image can be saved
cv2.imwrite('FinalEqualizedImage.jpg',finalEqualizedImage)
cv2.imshow('Airborne Equilized',finalEqualizedImage)
cv2.waitKey(0)



#finalEquilizedImage = np.zeros([airborne.shape[0],airborne.shape[1]])

#print "before applyin image  finalEquilizedImage [400][450] ", finalEquilizedImage[400][450]
#for i in range(equilizedImage.shape[0]):
#	for j in range(equilizedImage.shape[1]):
#		finalEquilizedImage[i][j] = int(equilizedImage[i][j])

#print "original image[400][450] ",airborne[400][450]
#print "FINAL equilizedImage[400][450] ",(finalEquilizedImage[400][450])

		
#print "original image[400][450] ",airborne[400][450]
#print "equilizedImage[400][450] ",(equilizedImage[400][450])

#print "original image[0][0] ",airborne[0][0]
#print "equilizedImage[0][0] ",(equilizedImage[0][0])

#print "original image[0][0] ",airborne[0][0]
#print "x[400][450] ",(x[400][450])

#print len(frequencyOfIntensity)

#def histogramFrequency(inputImage):
#	for intensity in range(255):
#		frequencyOfIntensity[intensity] = searchIntensityValues(inputImage,intensity)
		
#	print frequencyOfIntensity

#histogramFrequency(airborne)

		
