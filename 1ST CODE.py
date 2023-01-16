import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('elon_musk.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('bill_gates.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)



faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeElon],encodeTest)
facedis = face_recognition.face_distance([encodeElon],encodeTest)
print(result,facedis)

cv2.putText(imgTest,f'{result} {round(facedis[0],2)}',(25,25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

cv2.imshow('elon musk',imgElon)
cv2.imshow('elon test',imgTest)
cv2.waitKey(0)