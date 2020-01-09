# Face Detection and Recognition
Detection and recognition of a person in real time live video sequences using FaceNet-Keras and SVM classifier. Currently, working on optimizing the cpu usage.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. <br/>

### Prerequisites

```
OpenCV-Python libraries (CV2)
Keras - Deep learning framework
Numpy library
```
### Working Principle
1. Create a custom dataset of users/people. <br />
2. Extract face embeddings from the faces using pretrained keras facenet model in numpy-zip format. <br />
3. Train an SVM classifier using these face embeddings in (.npz) format, save the model using pickle.
4. Use this model to predict the face in a video sequence.  

## Running the Code

1. Create a dataset of the people involved in the "data/" folder. The hierarchy that should be followed is below: <br />
   data/ <br />
    </t>train/ <br />
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
   ```
   python3 dataset_prep.py
   ```

3. Run the extract_embeddings.py using python3.
   ```
   python3 extract_embeddings.py
   ```

4. Run the svm_classifier.py.
   ```
   python3 svm_classifier.py
   ```

5. Run webcam.py
   ```
   python3 webcam.py
   ```
   
### Demonstration
![Farmers Market Finder Demo](ScreenCapture_10-01-2020-00.42.46.gif)
   
