import cv2
import numpy as np
cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
filename = input('enter your name :')
dataset = "./data/"
offset = 20

#list to save images
facedata = []
skip =0
while True:
    success, img = cam.read()
    gryimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = model.detectMultiScale(gryimg,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])
    if len(faces) > 0:
        f = faces[-1]
        x,y,w,h = f
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,34,221),2)

        cropped_face = img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped_face = cv2.resize(cropped_face,(100,100))
        skip += 1
        if skip %10 == 0:
            facedata.append(cropped_face) 
            print("saved faces " + str(len(facedata)))
        cv2.imshow(filename,cropped_face)

    cv2.imshow('image window',img)
    
    key = cv2.waitKey(10) & 0xFF
    if(key == ord('q')):
        break

facedata = np.asarray(facedata)
m=facedata.shape[0]
facedata = facedata.reshape((m,-1))
file = dataset + filename + ".npy"
np.save( file,facedata)

cam.release()
cv2.destroyAllWindows()