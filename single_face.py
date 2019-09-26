import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import Normalizer


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
	return yhat

data = np.load('data.npz')
embed = np.load('5-celebrity-faces-embeddings.npz')
model = load_model('facenet_keras.h5')
test_face = data['arr_2'][0]
test_face_label = data['arr_3'][0]
test_embed = embed['arr_2'][0]

test_face_embedding = get_embedding(model,test_face)
in_encoder = Normalizer(norm='12')
test_x = in_encoder.transform(test_face_embedding)

test_face_embedding = in
samples = np.expand_dims(test_face_embedding, axis=0)
predicted_class = model.predict(samples)
predicted_prob = model.predict_proba(samples)

print(predicted_class)
print(predicted_prob)
