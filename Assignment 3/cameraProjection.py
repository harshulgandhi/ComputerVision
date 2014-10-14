import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pylab as pl


pts =  np.zeros([11,3])
#print pts 

u_o = 0
v_o = 0
beta_u = 1
beta_v = 1
k_u = 1
k_v = 1
focal_length =1


pts[0, :] = [-1, -1, -1]
pts[1, :] = [1, -1, -1]
pts[2, :] = [1, 1, -1]
pts[3, :] = [-1, 1, -1]
pts[4, :] = [-1, -1, 1]
pts[5, :] = [1, -1, 1]
pts[6, :] = [1,1, 1]
pts[7, :] = [-1, 1, 1]
pts[8, :] = [-0.5, -0.5, -1]
pts[9, :] = [0.5, -0.5, -1]
pts[10,:] = [0, 0.5, -1]

print pts

#quat1 = [0.0,1.0,-1.0,2.0]
#quat2 = [0.0,-1.0,-0.5,0.5]

#function to mulitple two quaternions.
def quaternionMultiplication(quat1,quat2):
	output = [0.0,0.0,0.0,0.0]

	output[0] = (quat1[0]*quat2[0]) - (quat1[1]*quat2[1] + quat1[2]*quat2[2] + quat1[3]*quat2[3])
 	output[1] = quat1[0]*quat2[1] + quat1[1]*quat2[0] + quat1[2]*quat2[3] - quat1[3]*quat2[2]
	output[2] = quat1[0]*quat2[2] - quat1[1]*quat2[3] + quat1[2]*quat2[0] + quat1[3]*quat2[1]
	output[3] = quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1] + quat1[3]*quat2[0]
	
	return output

#print quaternionMultiplication(quat1,quat2)	

def quaternionConjugate(quat):
	for i in range(1,len(quat)):
		quat[i] = -quat[i]
	
	return quat

#Camera's initial position given as (0,0,-5)
camFrame1Quat = [0,0,0,-5] 		#camera's initial position in quaternion form

def rotatePointAlongYaxis(pointQuat,theta):
	rotatedPointQuat = [0,0,0,0]
	rotationQuat =  [math.cos((math.radians(theta))/2), 0, math.sin((math.radians(theta))/2),0]
	
	rotatedPointQuat = quaternionMultiplication(quaternionMultiplication(rotationQuat,pointQuat),quaternionConjugate(rotationQuat))
#	print "Input quaternion : ",pointQuat
#	print "Angle : ",theta
#	print "(math.radians(theta))/2 ",(math.radians(theta))/2
#	print "quaternionConjugate(rotationQuat) : ",quaternionConjugate(rotationQuat)
	return rotatedPointQuat

camFrame2Quat = [0,0,0,0]
camFrame3Quat = [0,0,0,0]
camFrame4Quat = [0,0,0,0]
camFrame2Quat = rotatePointAlongYaxis(camFrame1Quat,-30)
camFrame3Quat = rotatePointAlongYaxis(camFrame2Quat,-30)
camFrame4Quat = rotatePointAlongYaxis(camFrame3Quat,-30)

print "FRAME1 : ", camFrame1Quat
print "Point after rotation1 - FRAME2 : ", camFrame2Quat
print "Point after rotation1 - FRAME3 : ", camFrame3Quat
print "Point after rotation1 - FRAME4 : ", camFrame4Quat

camFrameQuaternions = [camFrame1Quat,camFrame2Quat,camFrame3Quat,camFrame4Quat]

def quaternionToRotation(quat):
	rotationMatrix = np.zeros([3,3])
	rotationMatrix[0][0] = quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2 	
	rotationMatrix[0][1] = 2*(quat[1]*quat[2] - quat[0]*quat[3])
	rotationMatrix[0][2] = 2*(quat[1]*quat[3] + quat[0]*quat[2])
	rotationMatrix[1][0] = 2*(quat[1]*quat[2] + quat[0]*quat[3])
	rotationMatrix[1][1] = quat[0]**2 - quat[1]**2 + quat[2]**2 - quat[3]**2
	rotationMatrix[1][2] = 2*(quat[2]*quat[3] - quat[0]*quat[1])
	rotationMatrix[2][0] = 2*(quat[1]*quat[3] - quat[0]*quat[2]) 
	rotationMatrix[2][1] = 2*(quat[2]*quat[3] + quat[0]*quat[1])
	rotationMatrix[2][2] = quat[0]**2 - quat[1]**2 + quat[3]**2 - quat[2]**2

	return np.matrix(rotationMatrix)


quatIdentity = [0,1,1,1]
rotationQuat1 = [0,1,1,1]
rotation1_matrix = np.identity(3)
#rotation1_matrix = quaternionToRotation(rotationQuat1)
print "\nrotation matrix 1 :\n",rotation1_matrix
rotationQuat2 = [math.cos((math.radians(30))/2), 0, math.sin((math.radians(30))/2),0]
rotation2_matrix = np.zeros([3,3])
rotation2_matrix = quaternionToRotation(rotationQuat2)
print "\nrotation matrix 2 :\n",rotation2_matrix

 
rotationQuat3 = [math.cos((math.radians(60))/2), 0, math.sin((math.radians(60))/2),0]
rotation3_matrix = np.zeros([3,3])
rotation3_matrix = quaternionToRotation(rotationQuat3)
print "\nrotation matrix 3 :\n",rotation3_matrix


rotationQuat4 = [math.cos((math.radians(90))/2), 0, math.sin((math.radians(90))/2),0]
rotation4_matrix = np.zeros([3,3])
rotation4_matrix = quaternionToRotation(rotationQuat4)
print "\nrotation matrix 4 :\n",rotation4_matrix

 
rotationMatrices = [rotation1_matrix,rotation2_matrix,rotation3_matrix,rotation4_matrix]


### PERSPECTIVE PROJECTION ####
	
def pointPerspectiveProjection(point,cameraQuat,rotationMatrix):
	global focal_length
	global beta_u
        global beta_v
	global u_o
	global v_o
	
	
	Sp = np.zeros([3,1])		#this will hold point to be projected
	Sp[0][0] = point[0]		#Converting point quaternion into a 3x1 matrix as per formula
	Sp[1][0] = point[1]
	Sp[2][0] = point[2]
	
#	print "cameraQuat [1] = ",cameraQuat	
	Tf = np.zeros([3,1])		#this will hold the points for current frame of camera
	Tf[0][0] = cameraQuat[1]
	Tf[1][0] = cameraQuat[2]
	Tf[2][0] = cameraQuat[3]

	Sp_minus_Tf = Sp - Tf
	
#	orientationMatrixForCamera  = quaternionToRotation(cameraQuat)
#	print "\norientationMatrixForCamera[0][0] : ",orientationMatrixForCamera[0,0]
		
	If = np.zeros([3,1])		#this will hold camera's horizontal axis
#	print "\nIf[0][0] : ",If[0][0]
	
	If[0][0] = rotationMatrix[0,0]
	If[1][0] = rotationMatrix[0,1]
	If[2][0] = rotationMatrix[0,2]
	
	Jf = np.zeros([3,1])           #this will hold camera's horizontal axis
        Jf[0][0] = rotationMatrix[1,0]
        Jf[1][0] = rotationMatrix[1,1]
        Jf[2][0] = rotationMatrix[1,2]

	Kf = np.zeros([3,1])           #this will hold camera's horizontal axis
        Kf[0][0] = rotationMatrix[2,0]
        Kf[1][0] = rotationMatrix[2,1]
        Kf[2][0] = rotationMatrix[2,2]
	
	numerator = (focal_length*beta_u*np.dot((Sp_minus_Tf).transpose(),If))
	denominator = np.dot((Sp_minus_Tf).transpose(),Kf)
		
	Ufp= numerator/denominator
	#Ufp = U_matrixForm[0][0]
	Ufp = Ufp + u_o
	
	numerator = (focal_length*beta_v*np.dot((Sp_minus_Tf).transpose(),Jf))
	denominator = np.dot((Sp_minus_Tf).transpose(),Kf)
	Vfp = numerator/denominator 
	#Vfp = V_matrixForm[0][0]
        Vfp = Vfp + v_o


	return Ufp,Vfp


#print "Projection of first point : ",pointProjection(pts[0],camFrame1Quat)



#### ORTHOGRAPHIC PROJECTION ####

#function to give orthographic projection for a set of 3D points
def pointOrthographicProjection(point,cameraQuat,rotationMatrix):
	global focal_length
        global beta_u
        global beta_v
        global u_o
        global v_o


        Sp = np.zeros([3,1])            #this will hold point to be projected
        Sp[0][0] = point[0]             #Converting point quaternion into a 3x1 matrix as per formula
        Sp[1][0] = point[1]
        Sp[2][0] = point[2]

        #print "cameraQuat [1] = ",cameraQuat
        Tf = np.zeros([3,1])            #this will hold the points for current frame of camera
        Tf[0][0] = cameraQuat[1]
        Tf[1][0] = cameraQuat[2]
        Tf[2][0] = cameraQuat[3]

        Sp_minus_Tf = Sp - Tf

#       orientationMatrixForCamera  = quaternionToRotation(cameraQuat)
#       print "\norientationMatrixForCamera[0][0] : ",orientationMatrixForCamera[0,0]

        If = np.zeros([3,1])            #this will hold camera's horizontal axis
       # print "\nIf[0][0] : ",If[0][0]

        If[0][0] = rotationMatrix[0,0]
	If[1][0] = rotationMatrix[0,1]
        If[2][0] = rotationMatrix[0,2]

        Jf = np.zeros([3,1])           #this will hold camera's horizontal axis
        Jf[0][0] = rotationMatrix[1,0]
        Jf[1][0] = rotationMatrix[1,1]
        Jf[2][0] = rotationMatrix[1,2]

        Kf = np.zeros([3,1])           #this will hold camera's horizontal axis
        Kf[0][0] = rotationMatrix[2,0]
        Kf[1][0] = rotationMatrix[2,1]
        Kf[2][0] = rotationMatrix[2,2]


        U_matrixForm = (focal_length*beta_u*np.dot((Sp_minus_Tf).transpose(),If))
        Ufp = U_matrixForm[0][0]
        Ufp = Ufp + u_o


        V_matrixForm = (focal_length*beta_v*np.dot((Sp_minus_Tf).transpose(),Jf))
        Vfp = V_matrixForm[0][0]
        Vfp = Vfp + v_o


        return Ufp,Vfp

projectedModel = []

#function to get projection of all 11 points
def getProjectionForAllPoints(pts,cameraFrameQuat,rotationMatrices):
	global projectedModel
        projectedPointsF1 = np.zeros([11,2])
        projectedPointsF2 = np.zeros([11,2])
        projectedPointsF3 = np.zeros([11,2])
        projectedPointsF4 = np.zeros([11,2])
        projectedPointsF1Ortho = np.zeros([11,2])
        projectedPointsF2Ortho = np.zeros([11,2])
        projectedPointsF3Ortho = np.zeros([11,2])
        projectedPointsF4Ortho = np.zeros([11,2])

        for i in range(11):
               projectedPointsF1[i][0],projectedPointsF1[i][1] = pointPerspectiveProjection(pts[i],cameraFrameQuat[0],rotationMatrices[0])
	       projectedPointsF2[i][0],projectedPointsF2[i][1] = pointPerspectiveProjection(pts[i],cameraFrameQuat[1],rotationMatrices[1])
               projectedPointsF3[i][0],projectedPointsF3[i][1] = pointPerspectiveProjection(pts[i],cameraFrameQuat[2],rotationMatrices[2])
               projectedPointsF4[i][0],projectedPointsF4[i][1] = pointPerspectiveProjection(pts[i],cameraFrameQuat[3],rotationMatrices[3])
               projectedPointsF1Ortho[i][0],projectedPointsF1Ortho[i][1] = pointOrthographicProjection(pts[i],cameraFrameQuat[0],rotationMatrices[0])
               projectedPointsF2Ortho[i][0],projectedPointsF2Ortho[i][1] = pointOrthographicProjection(pts[i],cameraFrameQuat[1],rotationMatrices[1])
               projectedPointsF3Ortho[i][0],projectedPointsF3Ortho[i][1] = pointOrthographicProjection(pts[i],cameraFrameQuat[2],rotationMatrices[2])
               projectedPointsF4Ortho[i][0],projectedPointsF4Ortho[i][1] = pointOrthographicProjection(pts[i],cameraFrameQuat[3],rotationMatrices[3])


	
	projectedModel = [projectedPointsF1,projectedPointsF2,projectedPointsF3,projectedPointsF4,projectedPointsF1Ortho,projectedPointsF2Ortho,projectedPointsF3Ortho,projectedPointsF4Ortho]
	return projectedModel

print "Projections of all points w.r.t. fram1 : ",getProjectionForAllPoints(pts,camFrameQuaternions,rotationMatrices)
#print "\nProjected Model [0] : ",projectedModel[0][0][0]
                                                 

projectedModelPerspective = [projectedModel[0],projectedModel[1],projectedModel[2],projectedModel[3]]
projectedModelOrthogonal = [projectedModel[4],projectedModel[5],projectedModel[6],projectedModel[7]]
#print "\nprojectedModelPerspective[0]",projectedModelOrthogonal[0]
#plotSamplePoints(projectedModelPerspective,'Perspective')


import matplotlib.pyplot as plt1

x5 = []
y5 = []
x6 = []
y6 = []
x7 = []
y7 = []
x8 = []
y8 = []

## plotting orthographic projection
for i in range(len(projectedModel[4])):
        x5.append(projectedModel[4][i][0])
        y5.append(projectedModel[4][i][1])

plt1.subplot(221)
print "x5 :\n ",x5
print "y5 :\n ",y5
plt1.scatter(x5,y5)
plt1.title('Frame 1')

#       plt.plot(x1,y1)

for i in range(len(projectedModel[5])):
        x6.append(projectedModel[5][i][0])
        y6.append(projectedModel[5][i][1])

plt1.subplot(222)
plt1.scatter(x6,y6)
plt1.title('Frame 2')
#       plt.plot(x2,y2)

for i in range(len(projectedModel[6])):
        x7.append(projectedModel[6][i][0])
        y7.append(projectedModel[6][i][1])
plt1.subplot(223)
plt1.scatter(x7,y7)
plt1.title('Frame 3')

for i in range(len(projectedModel[7])):
        x8.append(projectedModel[7][i][0])
        y8.append(projectedModel[7][i][1])
plt1.subplot(224)
plt1.scatter(x8,y8)
plt1.title('Frame 4')
name = 'Orthographic.jpg'
plt1.savefig(name)
plt1.show()




x1 = []
y1 = []
x2 = []
y2 = []
x3 = []
y3 = []
x4 = []
y4 = []

## plotting perspective projection

for i in range(len(projectedModel[0])):
	x1.append(projectedModel[0][i][0])
	y1.append(projectedModel[0][i][1])

plt.subplot(2,2,1)
plt.scatter(x1,y1)
plt.title('Frame 1')

#       plt.plot(x1,y1)

for i in range(len(projectedModel[1])):
	x2.append(projectedModel[1][i][0])
	y2.append(projectedModel[1][i][1])

plt.subplot(2,2,2)
plt.scatter(x2,y2)
plt.title('Frame 2')
#       plt.plot(x2,y2)

for i in range(len(projectedModel[2])):
	x3.append(projectedModel[2][i][0])
	y3.append(projectedModel[2][i][1])
plt.subplot(2,2,3)
plt.scatter(x3,y3)
plt.title('Frame 3')

for i in range(len(projectedModel[3])):
	x4.append(projectedModel[3][i][0])
	y4.append(projectedModel[3][i][1])
plt.subplot(2,2,4)
plt.scatter(x4,y4)
plt.title('Frame 4')
name = 'Perspective.jpg'
plt.savefig(name)
plt.show()
#plotSamplePoints(projectedModelOrthogonal,'Orthographic')



