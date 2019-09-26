import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

    
cap = cv2.VideoCapture(0)
detector = MTCNN()
while True:
    ret, frame = cap.read()
    #faces = detect_face(frame)
    faces = detector.detect_faces(frame)
    for face in faces:
        (x,y,w,h) = face['box']
        face_pixels = frame[y:y+h,x:x+w]

        #Get word embedding for face
        get_word_embeddings(model,face_pixels)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
        cv2.imshow('Video',frame[y:y+h,x:x+w])
        

    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
