import cv2
import dlib
import numpy as np
import headpose_utils as headpose
from  utils import  *


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor.dat')
vid = cv2.VideoCapture(0)


while(True):
    
    ret, image = vid.read()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
    
        marks = predictor(gray, rect)
        marks_np = shape_to_np(marks)
        (x, y, w, h) = rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        distortion_coeff = np.zeros((4,1))
        h,w,c = headpose.getimageshape(image)
        imagePoints_2d = headpose.Imagepoints_2D_Matrix(marks)
        cameraMatrix = headpose.CameraMatrix(w, (h/2,w/2))
        successmsg , rotationVector , translationVector = headpose.SolvePnp(headpose.FaceModel_3D_Matrix(),
                    imagePoints_2d , cameraMatrix,distortion_coeff)
        
        # angles = headpose.CalcEulerAngles(rotationVector) #btl3 warning w error m3 enha zay lta7t bzbt
        
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        if angles[1] < -35:
            GAZE = "Looking: Left"
        elif angles[1] > 35:
            GAZE = "Looking: Right"
        else:
            GAZE = "Forward"
            
        cv2.putText(image, GAZE, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
        cv2.imshow("Head Pose", image)    


        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
vid.release()
cv2.destroyAllWindows()

