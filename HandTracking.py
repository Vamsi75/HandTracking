import cv2
import mediapipe as mp
import time
#readind video input from opencv
cap=cv2.VideoCapture(0)
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
#time for calculating framerate
pTime=0
cTime=0

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res=hands.process(imgRGB)
    #print(res.multi_hand_landmarks)  -to know landmarks
    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm) -- to know the landmark id's
                h, w, c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(cx,cy)
                #if id==0:
                cv2.circle(img,(cx,cy),10,(255,0,200),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)#for connecting landmarks

    cTime=time.time()
    fps=1/(cTime-pTime)#to find framerate
    pTime=cTime
    #displaying fps on screen
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,100),3)
    cv2.imshow("image",img)
    cv2.waitKey(1)
