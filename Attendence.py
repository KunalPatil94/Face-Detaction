import cv2
import numpy as np
import face_recognition as fr
import os
from datetime import datetime
path='face_images'
images=[]
classsname =[]
myList =os.listdir(path)
print(myList)

for cl in myList:
    curImg =cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classsname.append(os.path.splitext(cl)[0])
print(classsname)
 

def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAttendance(name):
    with open('Attain.csv','r+') as f:
        myDataList =f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')




encodeListKnown=findEncodings(images)
print('Encoding Complete')

cap= cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

    facescurFrame =fr.face_locations(imgs)
    encodecuFrame = fr.face_encodings(imgs,facescurFrame)

    for encodeFace,faceloc in zip(encodecuFrame,facescurFrame):
        matches=fr.compare_faces(encodeListKnown,encodeFace)
        faced=fr.face_distance(encodeListKnown,encodeFace)
        print(faced)
        matchIndex =np.argmin(faced)

        if matches[matchIndex]:
            name= classsname[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 =faceloc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('camera',img)
    cv2.waitKey(1)







# img = fr.load_image_file('face_images/Elon_Musk.jpg')
# imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
# imgTest =fr.load_image_file('face_images/Bill Gates.jpg')
# imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
# faceloc = fr.face_locations(imgElon)[0]
# encodeelon = fr.face_encodings(imgElon)[0]
# cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)


# facelocTest = fr.face_locations(imgTest)[0]
# encodetest = fr.face_encodings(imgTest)[0]
# cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)
