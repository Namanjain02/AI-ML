import cv2
import numpy as np
import matplotlib.pyplot as plit
from time import sleep
fd = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
sd = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

vid = cv2.VideoCapture(0)
notcaptured = True
seq = 0
while notcaptured:
    flag,img = vid.read()
    if flag:
        # processing code
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #  image ka color change karna ma help karta ha
        faces =fd.detectMultiScale(
            img_gray,
            scaleFactor =1.1,
            minNeighbors = 5,
            minSize = (50,50)
        )
        # smile =sd.detectMultiScale(
        #     img_gray,
        #     scaleFactor =1.1,
        #     minNeighbors = 5,
        #     minSize = (50,50)
        # )
        np.random.seed(50)
        colors = [np.random.randint(0,255,3).tolist()
                         for i in faces]
        i=0
        #   for smile  project
        # th, img_bw = cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY)
        for x,y,w,h in faces:
            face =img[y:y+h,x:x+w, :]
            smiles = sd.detectMultiScale(
                face,
                scaleFactor =1.1,
                minNeighbors = 5,
                minSize = (50,50)
            )
            if len(smiles)==1:
                seq +=1
                print(seq)
                if seq ==5:
                    cv2.imwrite('myselfie.png',img)
                    notcaptured = False
                    break
            else:
                seq = 0
            # x,y,w,h =(200,200,100,100) 
            # # img_crooped =  img[y:y+h,x:x+w, :]
            cv2.rectangle(
                img,pt1=(x,y), pt2 = (x+w,y+h),color=colors[i],
                thickness=8
            )
            i+=1
        cv2.imwrite('myselfie.png',img)
        cv2.imshow('preview',img)
        key = cv2.waitKey(1)
        if key == ord('q'):  # off karna ka liya key di gyi ha q name ki
            break
    else:
        print('No Frames')
        break
    sleep(0.1)
vid.release()
# cv2.waitKey(1)
cv2.destroyAllWindows()