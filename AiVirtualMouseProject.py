import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui


########################
wCam, hCam = 640, 480
#########################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


pTime = 0
detector = htm.handDetector(maxHands=1)
wSc, hSc = pyautogui.size()
frameR = 150 # frame redux
frameT = 120
smooth = 2
plocx, plocy = 0, 0
clocx, clocy = 0, 0

while True:
    #1. Find the hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)
        #2. Get the Tip of the imdex and middle fingers
        #3. Check fingers
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameT), (wCam - frameR, hCam - frameT), (255, 0, 255), 2)
        #4. Only index: moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            #5. convert cord

            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wSc))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hSc))

            #6. Smoothen Values
            clocx = plocx + (x3 - plocx) / smooth
            clocy = plocy + (y3 - plocy) / smooth

            #7. move mouse
            pyautogui.moveTo(wSc - clocx, clocy)
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocx, plocy = clocx, clocy


        #8. Both index and middle then clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10. Click mouse if distance short
            if length < 20:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

                pyautogui.click()


        #11. Frame Rate

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    #12. Display


    cv2.imshow("Image", img)
    cv2.waitKey(1)
