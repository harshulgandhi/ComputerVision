import numpy as np
import cv2
import math
pts =  np.zeros([11,3])
#print pts 

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

quat1 = [0.0,1.0,-1.0,2.0]
quat2 = [0.0,-1.0,-0.5,0.5]

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
	print "Input quaternion : ",pointQuat
	print "Angle : ",theta
	print "(math.radians(theta))/2 ",(math.radians(theta))/2
	print "quaternionConjugate(rotationQuat) : ",quaternionConjugate(rotationQuat)
	return rotatedPointQuat

camFrame2Quat = [0,0,0,0]
camFrame3Quat = [0,0,0,0]
camFrame4Quat = [0,0,0,0]
camFrame2Quat = rotatePointAlongYaxis(camFrame1Quat,30)
camFrame3Quat = rotatePointAlongYaxis(camFrame2Quat,30)
camFrame4Quat = rotatePointAlongYaxis(camFrame3Quat,30)

print "FRAME1 : ", camFrame1Quat
print "Point after rotation1 - FRAME2 : ", camFrame2Quat
print "Point after rotation1 - FRAME3 : ", camFrame3Quat
print "Point after rotation1 - FRAME4 : ", camFrame4Quat

def quaternionToRotation(quat):
	rotationMatrix = np.zero([3,3])
	rotationMatrix[0][0] = quat[0]**2 + quat[1]**2 - quat[2]**2 - quat[3]**2 	
	rotationMatrix[0][1] = 2(quat[1]*quat[2] - quat[0]*quat[3])
	rotationMatrix[0][2] = 2(quat[1]*quat[3] + quat[0]*quat[2])
	rotationMatrix[1][0] = 2(quat[1]*quat[2] + quat[0]*quat[3])
	rotationMatrix[1][1] = quat[0]**2 - quat[1]**2 + quat[2]**2 - quat[3]**2
	rotationMatrix[1][2] = 2(quat[2]*quat[3] - quat[0]*quat[1])
	rotationMatrix[2][0] = 2(quat[1]*quat[3] - quat[0]*quat[2]) 
	rotationMatrix[2][1] = 2(quat[2]*quat[3] + quat[0]*quat[1])
	rotationMatrix[2][2] = quat[0]**2 - quat[1]**2 + quat[3]**2 - quat[2]**2

	return np.matrix(rotationMatrix)







	
	



