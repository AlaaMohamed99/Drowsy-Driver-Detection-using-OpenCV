import numpy as np
# import dlib
import cv2

################ yarab mansash implement functions lel visualization############################

# 1- calibrate camera to get focal length
# 2-try γ=0
# 3-make visualization
# 4-lwindow msh bt2fl hata lw dost q (done)
# 5-check Euler angle
# 6-momken a deetct face wahed bs msh koloo(driver)

def HeadPoseAngles(marks,image,frame_counter_head):
    
        distortion_coeff = np.zeros((4,1))
        h,w,c = getimageshape(image)
        imagePoints_2d = Imagepoints_2D_Matrix(marks)
        cameraMatrix =CameraMatrix(w, (h/2,w/2))
        successmsg , rotationVector , translationVector = SolvePnp(FaceModel_3D_Matrix(),
                    imagePoints_2d , cameraMatrix,distortion_coeff)
        
        # angles = headpose.CalcEulerAngles(rotationVector) #btl3 warning w error m3 enha zay lta7t bzbt
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        max_no_of_frames = 35
        alarm = False
  
        if angles[1] < -10:
            frame_counter_head += 1
            if frame_counter_head >= max_no_of_frames:
                if not alarm:
                    alarm = True
                    GAZE = "Please Look Forward"
            
        elif angles[1] > 25:
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
#  get from the 68 facial landmarks from predictor(shape)

    imagePoints_2D = [[landmarks.part(30).x, landmarks.part(30).y],
                   [landmarks.part(8).x, landmarks.part(8).y],
                   [landmarks.part(36).x, landmarks.part(36).y],
                   [landmarks.part(45).x, landmarks.part(45).y],
                   [landmarks.part(48).x, landmarks.part(48).y],
                   [landmarks.part(54).x, landmarks.part(54).y]]
    
    imagePoints = np.array(imagePoints_2D, dtype=np.float64)
    return imagePoints     

def CameraMatrix(focal_l, center):
    #[[fx, γ , u0],     f(x, y) elfocal lenthgs bt3 lcamera , (u0,v0) elcenter of 2d image
    # [0, fy, v0,
    # [0, 0, 1]]
    
    cameraMatrix = [[focal_l, 1, center[0]],
                    [0, focal_l, center[1]],
                    [0, 0, 1]]
    
    
    return np.array(cameraMatrix, dtype=np.float)

def SolvePnp(facemodel_3D, imagePoints_2d, cameraMatrix, distortion_coeff):
# calculating rotation and translation vector using solvePnP
    successmsg, rotationVector, translationVector = cv2.solvePnP(facemodel_3D,imagePoints_2d, cameraMatrix, distortion_coeff)
    
    return successmsg , rotationVector , translationVector

def getimageshape(image):
    
    height, width, channels = image.shape
    
    return height , width , channels

# def GetCameraFocalLength(width):
#     focalLength = width    
#     return focalLength

def CalcEulerAngles(RotationVector):
# calculating angle

    RotationMatrix,jac = cv2.Rodrigues(RotationVector) #lazm ageebha matrix msh vector 
    three_angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(RotationMatrix)

    return three_angles
