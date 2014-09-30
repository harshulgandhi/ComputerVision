import numpy as np
import cv2
import cv2.cv as cv
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal
airborne = cv2.imread("airborne.jpg",0)
haze = cv2.imread("haze.jpg",0)
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


def cummulativeFrequency(histFrequency):	#calculates the cumulative frequency of each intensity value and returns a list
	cumulativeFrequency = [0]*256		#with cumulative frequency populated corresponding to each index representing
	for i in range(256):			#intensity value
		if(i==0):
			cumulativeFrequency[i] =  histFrequency[i]
		else:
			cumulativeFrequency[i] = cumulativeFrequency[i-1] + histFrequency[i]

	return cumulativeFrequency

equilizedImage = [[0 for x in xrange(960)] for x in xrange(720)]

def histEquilization(img):
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			equilizedImage[i][j] = (255*cumulativeFrequencyOfIntensity[img[i][j]])/(img.shape[0] * img.shape[1])
			
	return equilizedImage

####################### Histogram equalization for Airborne.jpg########################

plt.hist(imageMatrixToList(airborne),bins = 256) # to plot histogram
plt.title("Image histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

histogramFrequencyValues = histogramFrequency(airborne)

cumulativeFrequencyOfIntensity =  cummulativeFrequency(histogramFrequencyValues)

equilizedImage = histEquilization(airborne)
finalEqualizedImage = np.array(equilizedImage) # converting equalized 2D array into numpy array so that image can be saved
cv2.imwrite('FinalEqualizedImageAirborne.jpg',finalEqualizedImage)

plt.hist(imageMatrixToList(finalEqualizedImage),bins = 256) # to plot histogram
plt.title("Image histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

####################### Histogram equalization for Haze.jpg##########################

cv2.imwrite('Haze Grayscale.jpg',haze)

histogramFrequencyValues = histogramFrequency(haze)

cumulativeFrequencyOfIntensity =  cummulativeFrequency(histogramFrequencyValues)

equilizedImage = [[0 for x in xrange(700)] for x in xrange(525)]

plt.hist(imageMatrixToList(haze),bins = 256) # to plot histogram
plt.title("Image histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

equilizedImage = histEquilization(haze)
finalEqualizedImage = np.array(equilizedImage) # converting equalized 2D array into numpy array so that image can be saved
cv2.imwrite('FinalEqualizedImageHaze.jpg',finalEqualizedImage)


plt.hist(imageMatrixToList(finalEqualizedImage),bins = 256) # to plot histogram
plt.title("Image histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


	
