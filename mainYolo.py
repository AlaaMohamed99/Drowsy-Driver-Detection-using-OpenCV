import os 
import cv2
import time
import numpy as np
from PIL import Image
import tensorflow as tf
from phone.phone_detection import detect
from tensorflow.python.saved_model import tag_constants


model_path = os.path.join(os.getcwd() +'/phone/checkpoints/','yolov4_tiny_416')


def main(): 

    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    vid = cv2.VideoCapture(0)

    while True:
        _, frame = vid.read()
        result,phone_found = detect(frame,infer)  
        
        # create instance from your function 
        
        cv2.imshow("result", result)        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()




if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
