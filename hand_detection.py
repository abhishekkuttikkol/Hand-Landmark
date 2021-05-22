import cv2
import mediapipe as mp
import time
import numpy

cam = cv2.VideoCapture(0)
# cam.open('http://192.168.1.199:8080/video')

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=10, circle_radius=5)
pTime = 0
CTime = 0
# mageCanvas = numpy.zeros((720, 1280, 3), numpy.uint8)

while cam.isOpened():
    mageCanvas = numpy.zeros((720, 1280, 3), numpy.uint8)
    success,image = cam.read()
    imgRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(mageCanvas, handLMS, mpHands.HAND_CONNECTIONS, drawSpec, drawSpec)
            # mpDraw.draw_landmarks(image, handLMS, mpHands.HAND_CONNECTIONS, drawSpec, drawSpec)
            for id, lm in enumerate(handLMS.landmark):
                # print(id,lm)
                h, w, c = image.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id,cx,cy)
                
                if id == 4:
                    print(id,cx,cy)
                    # cv2.circle(image, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
    
    CTime = time.time()
    fps = 1/(CTime-pTime)
    pTime = CTime
    cv2.putText(image,str(int(fps)),(20,40),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("image",mageCanvas)
    cv2.imshow("image2",image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cam.release()