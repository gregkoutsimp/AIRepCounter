import cv2
import mediapipe as mp
import time
import math
import numpy as np

class poseDetector():

    def __init__(self, mode=False, upperBody = False, smooth =True,
                 detConf = 0.5, trackConf=0.5):

        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detConf = detConf
        self.trackConf = trackConf

        self.mpDraw = mp.solutions.drawing_utils
        #Initialize pose class
        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(self.mode, self.upperBody, self.smooth,
        #                              self.detConf, self.trackConf)
        self.pose = self.mpPose.Pose(self.mode, 2, self.smooth,
                                     self.detConf, self.trackConf)

    def findPose(self, img, draw = True):
        height, width, channel = img.shape

        m = height/256

        #Convert video to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(img)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def getPosition(self, img, draw=True):
            self.lmList = []
            if self.results.pose_landmarks:
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    height, width, channel = img.shape

                    # define pixel of landmarks
                    cx, cy = int(lm.x*width), int(lm.y*height)

                    self.lmList.append([id,cx,cy])

                    if draw:
                        cv2.circle(img, (cx, cy) , 5, (255,0,0), cv2.FILLED)
            return self.lmList

    def calc_angle(self,a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def getAngle(self, img,p1, p2, p3, draw = True):

        a = self.lmList[p1][1:]
        b = self.lmList[p2][1:]
        c = self.lmList[p3][1:]

        # Calc angle
        angle = self.calc_angle(a,b,c)

        # Draw landmarks
        if draw:

            cv2.line(img, (a[0], a[1]),(b[0], b[1]) ,(255, 255, 255), 3)
            cv2.line(img, (b[0], b[1]), (c[0], c[1]), (255, 255, 255), 3)
            cv2.circle(img, (a[0], a[1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (a[0], a[1]), 15, (255, 0, 0), 2)
            cv2.circle(img, (b[0], b[1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (b[0], b[1]), 15, (255, 0, 0), 2)
            cv2.circle(img, (c[0], c[1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (c[0], c[1]), 15, (255, 0, 0), 2)

            cv2.putText(img, str(int(angle)) , (b[0]+20, b[1]-50 ), cv2.FONT_HERSHEY_PLAIN,2,(0,0,255) ,2)

        return angle


    def bicepCurl(self, img, count,dir):
        img = self.findPose(img, False)
        lmList = self.getPosition(img, False)


        if len(lmList) != 0:
            # # Right arm
            angle_r = self.getAngle(img, 12, 14, 16)
            per_l = np.interp(angle_r, (45, 135), (0, 100))

            # Left arm
            angle_l = self.getAngle(img, 11, 13, 15)
            per_r = np.interp(angle_l, (45, 135), (0,100) )

            per = (per_l+per_r)/2
            #print(per)

            if per >= 90:
                if dir ==0:
                    count += 0.5
                    dir =1

            if per <= 10:
                if dir ==1:
                    count += 0.5
                    dir =0
            #print(count)

        return img, count,dir

    def squat(self, img, count,dir):
        img = self.findPose(img, False)
        lmList = self.getPosition(img, False)

        if len(lmList) != 0:

            #
            #Left Leg
            angle_l = self.getAngle(img, 23, 25, 27)
            per_l = np.interp(angle_l, (200, 280), (0, 100))

            #Right Leg
            angle_r = self.getAngle(img, 24, 26, 28)
            per_r = np.interp(angle_r, (200, 280), (0, 100))


            per = (angle_l+angle_r)/2
            #print(per)

            if per < 90:
                if dir ==0:
                    count += 0.5
                    dir =1

            if per > 90:
                if dir ==1:
                    count += 0.5
                    dir =0
            #print(count)

        return img, count,dir

    def pullUps(self, img, count,dir):
        img = self.findPose(img, False)
        lmList = self.getPosition(img, False)


        if len(lmList) != 0:
            # # Right arm
            angle_r = self.getAngle(img, 12, 14, 16)
            per_l = np.interp(angle_r, (45, 135), (0, 100))

            # Left arm
            angle_l = self.getAngle(img, 11, 13, 15)
            per_r = np.interp(angle_l, (45, 135), (0,100) )

            per = (per_l+per_r)/2
            #print(per)

            if per >= 90:
                if dir ==0:
                    count += 0.5
                    dir =1

            if per <= 10:
                if dir ==1:
                    count += 0.5
                    dir =0
            #print(count)

        return img, count,dir

    def boxCounter(self, img, count):

        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 16)

