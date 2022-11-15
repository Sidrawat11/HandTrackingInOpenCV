#####################################################################################
import cv2 as cv
# For FPS
import time
# NumPy for Interpolation of volume in percentage value
import numpy as np
# HandtrackingModule created before to make it easier
import HandTrackingMod as htm
# PyCaw to get volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#####################################################################################

# Controlling Camera windows size
wCam, hCam = 640, 480

# Capturing the Video from WebCam
capture = cv.VideoCapture(1)
capture.set(3, wCam)
capture.set(4, hCam)

# pTime for previousTime cTime for currentTime: Vars for FPS calculations
pTime = 0
cTime = 0

# PyCaw Given code to access volume functions of PC
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Getting the volume range usually from some -ve real number - to 0
volRange = volume.GetVolumeRange()

# Getting Max and Min Volumes
minVol = volRange[0]
maxVol = volRange[1]

# Volume Variables for volume controls
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 255, 255)
# Detector: hand detector class object to draw hand landmarks from mediapipe
# Detection Confidence set to 0.7 for smoother detection
detector = htm.HandDetector(detectionCon=0.7, maxHands=1)

# Showing Video
while True:
    isTrue, frame = capture.read()
    # Find hand in the Frame, draw set to false so that skeletal frame is not drawn between landmarks
    frame, _ = detector.findHands(frame)
    # Get the hand landmark list
    lmList, bbox = detector.findPosition(frame, label=True)
    # If there is hand in the frame then proceed
    if len(lmList) != 0:
        # 1 ---> Filter Based on Size
        area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) // 100
        if 250 <= area <= 1000:
            # Find Distance Between Index and thumb
            length, frame, lineInfo = detector.findDist(4, 8, frame)

            # Calculate the volume Bar length and percentage
            volBar = np.interp(length, [30, 210], [400, 50])
            volPer = np.interp(length, [30, 210], [0, 100])

            # Making it a smoother
            smoothness = 5
            volPer = smoothness * round(volPer / smoothness)
            # Check Fingers Up
            fingers = detector.fingersUp()
            if not fingers[4]:
                # Setting the volume of the PC accordingly
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                cv.circle(
                    frame, (lineInfo[4], lineInfo[5]), 5, (0, 255, 255), cv.FILLED)
                colorVol = (0, 255, 0)

    # Drawing the volume bar
    cv.rectangle(frame, (50, 50), (95, 400), (255, 255, 255), 2)
    cv.rectangle(frame, (50, int(volBar)), (95, 400),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, f'{(int(volPer))}%', (45, 425),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    if cVol == 0:
        volume.SetMute(1, None)
        cv.putText(frame, f'Vol Set: {int(cVol)} Mute', (450, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, colorVol, 2)
    else:
        volume.SetMute(0, None)
        cv.putText(frame, f'Vol Set: {int(cVol)}', (450, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.65, colorVol, 2)

        # Putting the FPS on the Screen
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, "FPS:" + str(int(fps)), (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    # Show the Frame
    cv.imshow('Video', frame)

    # Closing the frame when 'SpaceBar' is entered
    if cv.waitKey(1) & 0xFF == ord(' '):
        break

# Releasing the capture object and destroying all windows
capture.release
cv.destroyAllWindows()
