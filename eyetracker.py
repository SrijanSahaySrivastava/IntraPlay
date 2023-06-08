import math
import cv2
import mediapipe as mp
import time
import numpy as np
map_face_mesh = mp.solutions.face_mesh

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

class eyeTracker():
    def __init__(self,detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpFaceMesh = map_face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        
        self.frame_counter =0
        self.CEF_COUNTER =0
        self.TOTAL_BLINKS =0
        # constants
        self.CLOSED_EYES_FRAME =3
        self.FONTS =cv2.FONT_HERSHEY_COMPLEX

        # face bounder indices 
        self.FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

        # lips indices for Landmarks
        self.LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
        self.LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
        # Left eyes indices 
        self.LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        self.LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

        # right eyes indices
        self.RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
        self.RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
        self.start = 0
        self.end = 0
        
    def landmarksDetection(self, img, results, draw=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        if draw :
            [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
        
        return mesh_coord
    
    def blinkRatio(self, img, landmarks, right_indices, left_indices):
        # Right eyes 
        # horizontal line 
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line 
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes 
        # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
        # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

        # LEFT_EYE 
        # horizontal line 
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]

        # vertical line 
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]

        rhDistance = euclaideanDistance(rh_right, rh_left)
        rvDistance = euclaideanDistance(rv_top, rv_bottom)

        lvDistance = euclaideanDistance(lv_top, lv_bottom)
        lhDistance = euclaideanDistance(lh_right, lh_left)

        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance

        ratio = (reRatio+leRatio)/2
        return ratio 

    def eyesExtractor(self, img, right_eye_coords, left_eye_coords):
        # converting color image to  scale image 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # getting the dimension of image 
        dim = gray.shape

        # creating mask from gray scale dim
        mask = np.zeros(dim, dtype=np.uint8)

        # drawing Eyes Shape on mask with white color 
        cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
        cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

        # showing the mask 
        # cv.imshow('mask', mask)
        
        # draw eyes image on mask, where white shape is 
        eyes = cv2.bitwise_and(gray, gray, mask=mask)
        # change black color to gray other than eys 
        # cv.imshow('eyes draw', eyes)
        eyes[mask==0]=155
        
        # getting minium and maximum x and y  for right and left eyes 
        # For Right Eye 
        r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
        r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
        r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
        r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

        # For LEFT Eye
        l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
        l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
        l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
        l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

        # croping the eyes from mask 
        cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
        cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

        # returning the cropped eyes 
        return cropped_right, cropped_left
    
    def positionEstimator(self, cropped_eye):
        # getting height and width of eye 
        h, w =cropped_eye.shape
        
        # remove the noise from images
        gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)
        median_blur = cv2.medianBlur(gaussain_blur, 3)

        # applying thrsholding to convert binary_image
        ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

        # create fixd part for eye with 
        piece = int(w/3) 

        # slicing the eyes into three parts 
        right_piece = threshed_eye[0:h, 0:piece]
        center_piece = threshed_eye[0:h, piece: piece+piece]
        left_piece = threshed_eye[0:h, piece +piece:w]
        
        # calling pixel counter function
        eye_position = pixelCounter(right_piece, center_piece, left_piece)

        return eye_position