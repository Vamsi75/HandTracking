import cv2
import mediapipe as mp
import time

class HandDetector():
    def __init__(self,mode=False, maxHands=2, detconf=0.5, trackconf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detconf = detconf
        self.trackconf = trackconf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res=self.hands.process(imgRGB)
        #print(res.multi_hand_landmarks)  -to know landmarks
        if self.res.multi_hand_landmarks:
            for handLms in self.res.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)#for connecting landmarks
        return img
    def FindPosition(self, img, handNo=0, draw=True):
        lmList=[]
        if self.res.multi_hand_landmarks:
            myHand = self.res.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(cx,cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255,0,200),cv2.FILLED)
        return lmlist
def main():
    pTime=0
    cTime=0
    detector = HandDetector()
    cap=cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img=detector.findHands(img)
        lmList=detector.FindPosition(img)
        if len(lmList)!=0:
            print(lmList[4])

        cTime=time.time()
        fps=1/(cTime-pTime)#to find framerate
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,100),3)

        cv2.imshow("image",img)
        cv2.waitKey(1)
if __name__=="__main__":
    main()
