import cv2
import mediapipe as mp
import time
import math
import numpy as np
# from videoplayer import VideoPlayer
import tkinter as tk
import pyautogui

# player = VideoPlayer(tk.Tk())
#Volume Control by PyCAW
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
map_face_mesh = mp.solutions.face_mesh
# variables 
from eyetracker import eyeTracker
from handTrack import handTracker


frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
FONTS =cv2.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
    elif max_index==1:
        pos_eye = 'CENTER'
    elif max_index ==2:
        pos_eye = 'LEFT'
    else:
        pos_eye="Closed"
    return pos_eye


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
def VideoPlay(player, is_playing):

    # player.root.mainloop()
    is_playing = 1
    
    print("Video Play")

def VideoPause():
    pyautogui.press('space')
    print("Video Pause")

def VideoStop():
    print("Video Stop")

def VideoForward():
    print("Video Forward")
#---------------------------------------------------------------


#---------------------------------------------------------------
#Main Function
#---------------------------------------------------------------
def main():
    is_playing = 0
    with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
        
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)
        tracker = handTracker(maxHands=1, detectionCon=0.8)
        eye = eyeTracker()
        
        while True:
            success,image = cap.read()
            image = tracker.handsFinder(image)
            lmList = tracker.positionFinder(image)
            volumeRange = volume.GetVolumeRange()
            minVol = volumeRange[0]
            maxVol = volumeRange[1]
            
            #Eye Position Tracker------------------------------------------
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results  = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_coords = eye.landmarksDetection(image, results, False)
                ratio = eye.blinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
                if ratio >5.5:
                    eye.CEF_COUNTER +=1
                else:
                    if eye.CEF_COUNTER>eye.CLOSED_EYES_FRAME:
                        eye.TOTAL_BLINKS +=1
                        eye.CEF_COUNTER =0
                
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eye.eyesExtractor(image, right_coords, left_coords)
                eye_position = eye.positionEstimator(crop_right)
                eye_position_left= eye.positionEstimator(crop_left)
                #print(eye_position, eye_position_left)
                if eye_position != "CENTER" and eye_position_left != "CENTER" and is_playing == 1:
                    pyautogui.press('space')
                    print("Space Pressed pause")
                    is_playing = 0
                if eye_position == "CENTER" and eye_position_left == "CENTER" and is_playing == 0:
                    pyautogui.press('space')
                    print("Space Pressed play")
                    is_playing = 1
                
            elif is_playing == 1:
                print("No Face Found")
                pyautogui.press('space')
                print("Space Pressed pause")
                is_playing = 0
            #----------------------------------------------------------------
                
            #print(ratio, mesh_coords)
            if len(lmList) != 0:
                #print(len(lmList))         #21 points per hand
                
                #Fingers Up-------------------------------------------------------
                fingers = tracker.fingersUp(lmList)
                #print(fingers)
                #----------------------------------------------------------------
                
                if fingers[0]==1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 1:
                    pyautogui.press('space')
                    time.sleep(0.5)

                if fingers[1] ==0 and fingers[2]== 0 and fingers[3] ==0 and fingers[0] == 0:
                    pyautogui.press('f')
                    time.sleep(0.5)
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