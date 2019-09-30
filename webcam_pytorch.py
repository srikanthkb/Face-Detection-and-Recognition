from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

mtcnn = MTCNN(image_size=160)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

image = Image.open('data\\train\\ben_afflek\httpssmediacacheakpinimgcomxeebdfdbaaajpg.jpg')

image_cropped = mtcnn(image, save_path='test.jpg')
#image_embedding = resnet(image_cropped.unsqueeze(0))

resnet.classify = True
image_probs = resnet(image_cropped.unsqueeze(0))
