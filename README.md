# Face Detection and Recognition
Detection and recognition of a person in real time live video sequences.
How to run:
1. Create a dataset of the people involved in the "data/" folder. The hierarchy that should be followed is below: <br />
   data/ <br />
    </t>strain/ <br />
      </t> </t> person1/ <br />
      </t> </t> </t>  image1<br />
      </t></t></t>  image2 <br />
      </t></t></t>  (...) <br />
      </t></t></t> .<br />
      </t></t></t> .<br />
      </t></t></t> person*n/<br />
    </t> val/<br />
      </t></t> person1/<br />
      </t></t></t> (...)<br />
      </t></t></t> .<br />
      </t></t></t>.<br />
      </t></t></t> person*n/<br />

2. Run the dataset_prep.py program using python3.
   This creates a dataset of all the faces detected in the images stored in data/ folder. This data of all faces is saved onto the data.npz numpy zip file. The data is stored in a format, that allows us to extract the labels easily.

3. Run the extract_embeddings.py using python3.
   This extracts the embeddings for all the faces stored in data.npz using a pretrained facenet model in keras.
   These embeddings are again saved onto face_embeddings.npz numpy zip file, with embeddings and labels correspondingly.

4. Run the svm_classifier.py.
   SVM classisfier is trained on the word embeddings and the labels stored in face_embeddings.npz and the model weights are dumped using 
   pickle

5. Run webcam.py
   This program uses live feed from webcam and detects and recognizes the faces(if any) in the frame, in a continous loop.
   
   
