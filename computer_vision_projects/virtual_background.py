##author - susant achary
import cv2
import time
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


cap = cv2.VideoCapture(0) # for realtime
cap.set(3, 1280)
cap.set(4, 720)

BG_COLOR = (255,255,0)

prev_time = 0

with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    bg_image = None
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue


        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True

        condition = np.stack((results.segmentation_mask,) * 3, axis = -1) > 0.1

        #apply background 
        bg_image = cv2.imread(r'D:\Work\kaggle\cassava\pexels-photo-3968061.jpeg')
        #bg_image = cv2.GaussianBlur(image,(55,55),0)

        if bg_image is None:
            bg_image = np.zeros(image.shape,dtype = np.uint8)
            bg_image[:] = BG_COLOR
        out_image = np.where(condition, image, bg_image)
        
        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(out_image,f'FPS: {int(fps)}',(20,70), cv2.FONT_HERSHEY_PLAIN,3,(0,192,255),2)


        cv2.imshow("Video Call Background", out_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
