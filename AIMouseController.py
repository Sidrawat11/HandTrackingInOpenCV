import cv2 as cv
import numpy as np
import time
import pyautogui
import pyautogui as pg
import HandTrackingMod as htm

#Creating Variables to Set the Size of the windows
wCam, hCam = 640, 480
wScreen, hScreen = pg.size()
smoothening = 7

#Reduce Frame
frameR = 100

#Capture Video from Camera
capture =  cv.VideoCapture(1)
capture.set(3, wCam)
capture.set(4, hCam)

# Frame Rate Values
cTime = 0
pTime = 0

#Varaibles to Smoothen the value
prevX, prevY = 0, 0
currX, currY = 0, 0

detector = htm.HandDetector(maxHands=1,detectionCon=0.7)

#Showing the Video
while True:
    isTrue, frame = capture.read()
    _,  number_of_hands = detector.findHands(frame, True)
    # Get the Tip of index and middle fingers
    lmList, bbox = detector.findPosition(frame, draw=True)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        cv.rectangle(frame, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)
        # Check which fingers are up
        fingers = detector.fingersUp()
        # Only Index Finger :  Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Convert Coordinates
            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScreen))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScreen))
            # Smoothen values
            currX = prevX + (x3 - prevX)/ smoothening
            currY = prevY + (y3 - prevY)/ smoothening
            # Move Mouse
            pg.moveTo(wScreen-currX, currY)
            cv.circle(frame, (x1,y1),10, (255,0,255), cv.FILLED)
            prevX, prevY = currX, currY
        # Checking if it's clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, frame,lineInfo = detector.findDist(8,12, frame)
            if length < 25:
                cv.circle(frame, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv.FILLED)
                pg.click()

    # FPS Calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(frame, "FPS:" + str(int(fps)), (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    cv.imshow('Video',frame)
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

capture.release
cv.destroyAllWindows()

