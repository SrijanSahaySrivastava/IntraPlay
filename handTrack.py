import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False, maxHands=2, detectionConfi = 0.5, TrackCOnfi=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConfi=detectionConfi
        self.TrackCOnfi=TrackCOnfi
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConfi,self.TrackCOnfi)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img       
        
    def findPosition(self,img,handNo=0,draw=True):
        mList= []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    mList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return mList