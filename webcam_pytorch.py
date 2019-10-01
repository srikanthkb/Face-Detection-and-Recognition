import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1

#Classes
people = ['Ben Affleck', 'Elton John', 'Jerry Seinfeld', 'Madonna', 'Mindy Kaling', 'Srikanth Kb']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#mtcnn
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True
)
#resnet
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
model = pickle.load(open('svm_model_pytorch.sav','rb'))
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #faces = detect_face(frame)
    faces = mtcnn(frame)
    for face in faces:
        (x,y,w,h) = face['box']
        face_pixels = frame[y:y+h,x:x+w]

        #Get word embedding for face
        #face_embedding = resnet(face_pixels_resized)

        
        #predict from SVM model
        #predicted_label = model.predict(face_embedding)
        #predicted_name  = people[predicted_label[0]]
        
        #Draw rectangle on face and name
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
        #cv2.putText(frame,predicted_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1, cv2.LINE_AA)
    cv2.imshow('Video',frame)
        

    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
