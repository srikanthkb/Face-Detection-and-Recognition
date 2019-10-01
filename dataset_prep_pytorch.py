from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

def extract_face(file,required_size=(160,160)):
    image = Image.open(file)
    print('Working on image:',file)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    try:
        x,y,w,h = results[0]['box']
    except:
        print('face not found:')
    x,y = abs(x),abs(y)
    face = pixels[y:y+h,x:x+w]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array

def load_images(directory):
    faces = list()
    for filename in os.listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

def load_dataset(directory):
    X, y = list(), []
    for subdir in os.listdir(directory):
        path = directory+subdir+'\\'
        if not os.path.isdir(path):
            continue
        print('Working on directory: ',subdir)
        faces = load_images(path)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)

train_dataset, train_labels = load_dataset('data\\train\\')
val_dataset, val_labels = load_dataset('data\\val\\')
