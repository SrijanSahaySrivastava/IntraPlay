import cv2
import mediapipe as mp
import time
import math
import numpy as np
#Volume Control by PyCAW
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#----------------------------------------------------------------
#Hand Tracking Module
#----------------------------------------------------------------
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def handsFinder(self,image,draw=True):
        imageRGB = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def fingersUp(self, lmList):
        fingers = []
        fingerTips= [8,12,16,20]
        for tip in fingerTips:
            if lmList[tip][2] < lmList[tip-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist
#----------------------------------------------------------------

#----------------------------------------------------------------
#Volume Control Module
#----------------------------------------------------------------
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
#volume.GetMute()
#---------------------------------------------------------------


#---------------------------------------------------------------
#Video Control
#---------------------------------------------------------------
def VideoPlay():
    pass

def VideoPause():
    pass

def VideoStop():
    pass

def VideoForward():
    pass
#---------------------------------------------------------------


#---------------------------------------------------------------
#Main Function
#---------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    tracker = handTracker(maxHands=1, detectionCon=0.8)

    while True:
        success,image = cap.read()
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        volumeRange = volume.GetVolumeRange()
        minVol = volumeRange[0]
        maxVol = volumeRange[1]
        if len(lmList) != 0:
            #print(len(lmList))         #21 points per hand
            
            #Fingers Up-------------------------------------------------------
            fingers = tracker.fingersUp(lmList)
            print(fingers)
            #----------------------------------------------------------------
            
            
            
            
            #Volume Control-----------------------------------------------------
            xp, yp = lmList[8][1], lmList[8][2]
            xpinky_tip, ypinky_tip = lmList[20][1], lmList[20][2]
            xpinky_mcp, ypinky_mcp = lmList[17][1], lmList[17][2]
            xt,yt = lmList[4][1], lmList[4][2]
            length1 = math.hypot(xt-xp,yt-yp)
            #length2 = math.hypot(xpinky_tip-xpinky_mcp,ypinky_tip-ypinky_mcp)
            cv2.circle(image,(xp,yp), 15 , (255,0,255), cv2.FILLED)
            cv2.circle(image,(xt,yt), 15 , (255,0,255), cv2.FILLED)
            vol = np.interp(length1,[50,143],[minVol,maxVol])
            if fingers[3] == 0 and fingers[2] == 0 and fingers[1] == 0:
                volume.SetMasterVolumeLevel(vol, None)
                cv2.circle(image,(xp,yp), 15 , (0,255,0), cv2.FILLED)
                cv2.circle(image,(xt,yt), 15 , (0,255,0), cv2.FILLED)
                
            # xMFT, yMFT = lmList[12][1], lmList[12][2]
            # xW, yW = lmList[0][1], lmList[0][2]
            # length2 = math.hypot(xW-xMFT,yW-yMFT)
            # print(length2)
            # if length2 < 80:
            #     volume.GetMute()
            #-------------------------------------------------------------------
                

        cv2.imshow("Video",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
#----------------------------------------------------------------


if __name__ == "__main__":
    main()