###################################################################
# This a Module for HandTracking so the code can be reused without
# typing everything again.
###################################################################

# Importing Libraries
import cv2 as cv
import mediapipe as mp
import math


# Function to put rectangle around a text
def putRect(frame, label, x, y, color=(0, 0, 0), font_thickness=1):
    (w, h), _ = cv.getTextSize(
        label, cv.FONT_HERSHEY_COMPLEX, 0.9, font_thickness)
    cv.rectangle(frame, (x - 21, y - 60),
                 (x - 12 + w, y - 20), (0, 255, 0), -1)
    cv.putText(frame, label, (x - 12, y - 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.9, color, font_thickness)


class HandDetector:
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.results = None
        self.lmList = None
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.handDirection = ""

    # Function to find hands on the frame and draw a skeletal frame through the landmarks
    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        numHands = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                numHands = numHands + 1
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame, numHands

    # Function to return the list of landmarks on the Hand,
    # According to mediapipe there are total of 21 landmarks on the hand
    def findPosition(self, frame, handNumber=0, draw=True, label=False):
        xList = []
        yList = []
        bound_box = []
        # Landmark List to return
        self.lmList = []
        # Find the hand on the screen and add there position onto the list
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]
            for ID, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([ID, cx, cy])
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bound_box = xmin, ymin, xmax, ymax

            if draw:
                cv.rectangle(frame, (bound_box[0] - 20, bound_box[1] - 20),
                             (bound_box[2] + 20, bound_box[3] + 20), (0, 255, 0), 3)

                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[4]][1]:
                    self.handDirection = "Left"
                    if label:
                        putRect(frame, self.handDirection,
                                bound_box[0], bound_box[1], (255, 255, 255), 2)
                else:
                    self.handDirection = "Right"
                    if label:
                        putRect(frame, self.handDirection,
                                bound_box[0], bound_box[1], (255, 255, 255), 2)

        return self.lmList, bound_box

    # Function to check if the fingers are up
    # This is done by comparing positions of landmark of the finger
    def fingersUp(self):
        finger = []
        # Checking which hand it is left or right
        # For Thumb
        if self.handDirection == "Left":
            if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                finger.append(0)
            else:
                finger.append(1)
        else:
            if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                finger.append(0)
            else:
                finger.append(1)

        # For 4 fingers
        for fingerId in range(1, 5):
            if self.lmList[self.tipIds[fingerId]][2] < self.lmList[self.tipIds[fingerId] - 2][2]:
                finger.append(1)
            else:
                finger.append(0)
        return finger

    def findDist(self, p1, p2, frame, draw=True):
        # Getting coordinates of the thumb and index finger respectively
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        # Getting the center point between the two points
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            # Drawing circles at the extremities and the center, while drawing a line though them
            cv.circle(frame, (x1, y1), 10, (0, 0, 0), cv.FILLED)
            cv.circle(frame, (x2, y2), 10, (0, 0, 0), cv.FILLED)
            cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

        # Get the length between two points on the finger
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]

# def main():
#     pTime = 0
#     cTime = 0
#     capture = cv.VideoCapture(0)
#     detector = handDetector()
#     while True:
#         isTrue, frame = capture.read()
#         frame = detector.findHands(frame)
#         lmList = detector.findPosition(frame)
#         if cv.waitKey(20) & 0xFF == ord('d'):
#             break
#     capture.release
#     cv.destroyAllWindows()


# if __name__ == "__main__":
#     main()
