import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

#Classes
people = ['Ben Affleck', 'Elton John', 'Jerry Seinfeld', 'Madonna', 'Mindy Kaling', 'Srikanth Kb']

cap = cv2.VideoCapture(0)
detector = MTCNN()
keras_model = load_model('facenet_keras.h5')

#Normalizer
encoder = Normalizer(norm='l2')

#SVM model
model = pickle.load(open('svm_model.sav','rb'))


while True:
    ret, frame = cap.read()
    #faces = detect_face(frame)
    faces = detector.detect_faces(frame)
    for face in faces:
        (x,y,w,h) = face['box']
        face_pixels = frame[y:y+h,x:x+w]

        #resize to (160,160)
        face_pixels_resized = cv2.resize(face_pixels,(160,160), interpolation=cv2.INTER_AREA)

        #Get word embedding for face
        face_embedding = get_embedding(keras_model,face_pixels_resized)

        #normalize the embedding
        face_embedding = encoder.transform(face_embedding.reshape(1,-1))

        #predict from SVM model
        predicted_label = model.predict(face_embedding)
        predicted_name  = people[predicted_label[0]]
        
        #Draw rectangle on face and name
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
        cv2.putText(frame,predicted_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1, cv2.LINE_AA)
    cv2.imshow('Video',frame)
        

    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
