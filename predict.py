from PIL import Image, ImageOps
import math
from main import model

def preprocess(path):
    img = Image.open(path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    pixels = list(img.getdata())
    pixels = [p/255 for p in pixels]

    return pixels
    

if __name__ == '__main__':
    img_path = 'handwritten9.jpeg'
    x = preprocess(img_path)
    prediction = model.knnModel(x)
    print(prediction)