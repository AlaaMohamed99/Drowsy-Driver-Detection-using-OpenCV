import numpy as np
# import dlib
import cv2


def HeadPoseAngles(marks,image,frame_counter_head):
        #marks : 68 facial landmarks 
        #frame_counter_head : counter for num of frames to detect if the driver looked away to turn alarm on
        distortion_coeff = np.zeros((4,1)) #no camera length distortion
        h,w,c = getimageshape(image)
        imagePoints_2d = Imagepoints_2D_Matrix(marks)
        cameraMatrix =CameraMatrix(w, (h/2,w/2))
        #head pose estimation using solvepnp using iterative algorithm
        #estimate the orientation(projection) of a 3D object in a 2D image
        successmsg , rotationVector , translationVector = SolvePnp(FaceModel_3D_Matrix(),
                    imagePoints_2d , cameraMatrix,distortion_coeff)
        
        rmat, jac = cv2.Rodrigues(rotationVector) #must change from vector to matrix
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        max_no_of_frames = 45 #num of frames to detect if the driver looked away to turn alarm on
        alarm = False
  
        if angles[1] < -10: #looking left
            frame_counter_head += 1
            if frame_counter_head >= max_no_of_frames:
                if not alarm:
                    alarm = True
                    GAZE = "Please Look Forward"
            
        elif angles[1] > 25: #looking right
            frame_counter_head += 1
            if frame_counter_head >= max_no_of_frames:
                if not alarm:
                    alarm = True            
                    GAZE = "Please Look Forward"
                    
        
        else:
            frame_counter_head = 0        
            GAZE = "............ "
            alarm = False
            
        return angles , alarm , frame_counter_head

def FaceModel_3D_Matrix():
#[x y z 1] 3d model of face  (World Coordinates)  
#3D location of the 2D feature image points
    modelPoints_3D = [[0.0, 0.0, 0.0], # Nose tip
                   [0.0, -330.0, -65.0],# Chin
                   [-225.0, 170.0, -135.0],# Left eye left corner
                   [225.0, 170.0, -135.0],# Right eye right corner
                   [-150.0, -150.0, -125.0],# Left Mouth corner
                   [150.0, -150.0, -125.0]] # Right mouth corner
    
    
    modelpoints = np.array(modelPoints_3D, dtype=np.float64)
    
    return modelpoints

def Imagepoints_2D_Matrix(landmarks):
#s[u v t] 2d image taken by camera
#  get from the 68 facial landmarks from predictor (from dlib library )(shape) the x,y coordinates for some landmarks

    imagePoints_2D = [[landmarks.part(30).x, landmarks.part(30).y], #Nose Tip
                   [landmarks.part(8).x, landmarks.part(8).y], #Chin
                   [landmarks.part(36).x, landmarks.part(36).y], #Left corner of the left eye
                   [landmarks.part(45).x, landmarks.part(45).y], #Right corner of the right eye
                   [landmarks.part(48).x, landmarks.part(48).y], #Left corner of the mouth
                   [landmarks.part(54).x, landmarks.part(54).y]] #Right corner of the mouth
    
    imagePoints = np.array(imagePoints_2D, dtype=np.float64)
    return imagePoints     

def CameraMatrix(focal_l, center):
    #[[fx, γ , u0],     f(x, y) elfocal lenthgs bt3 lcamera , (u0,v0) elcenter of 2d image
    # [0, fy, v0,
    # [0, 0, 1]]
    #calibrating camera    
    cameraMatrix = [[focal_l, 1, center[0]],
                    [0, focal_l, center[1]],
                    [0, 0, 1]]
    
    
    return np.array(cameraMatrix, dtype=np.float)

def SolvePnp(facemodel_3D, imagePoints_2d, cameraMatrix, distortion_coeff):
# calculating rotation and translation vector using solvePnP
    successmsg, rotationVector, translationVector = cv2.solvePnP(facemodel_3D,imagePoints_2d, cameraMatrix, distortion_coeff)
    
    return successmsg , rotationVector , translationVector

def getimageshape(image):
        #get the h,w,c of image
    height, width, channels = image.shape
    
    return height , width , channels


def CalcEulerAngles(RotationVector):
# calculating angle

    RotationMatrix,jac = cv2.Rodrigues(RotationVector) #lazm ageebha matrix msh vector 
    three_angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(RotationMatrix)

    return three_angles
