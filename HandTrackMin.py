import cv2 as cv
import mediapipe as mp
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    isTrue, frame = capture.read()
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w , c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 4:
                    cv.circle(frame, (cx,cy), 25, (255,0,0), cv.FILLED)
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame, "FPS:"+str(int(fps)), (10,70), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    cv.imshow('Image',frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release
cv.destroyAllWindows()

