import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


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


    
cap = cv2.VideoCapture(0)
detector = MTCNN()
keras_model = load_model('facenet_keras.h5')

#Normalizer
encoder = Normalize(norm='l2')

#SVM model
model = SVC(kernel='linear', probability=True)


while True:
    ret, frame = cap.read()
    #faces = detect_face(frame)
    faces = detector.detect_faces(frame)
    for face in faces:
        (x,y,w,h) = face['box']
        face_pixels = frame[y:y+h,x:x+w]

        #Get word embedding for face
        face_embedding = get_word_embeddings(keras_model,face_pixels)

        #normalize the embedding
        face_embedding = encoder.transform(face_embedding.reshape(1,-1))

        #predict from SVM model
        
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)
        cv2.imshow('Video',frame[y:y+h,x:x+w])
        

    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
