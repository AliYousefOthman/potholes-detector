from ultralytics import YOLO
from PIL import Image
import uvicorn


def predict_img(img,model):
    result = model.predict(img)

    result = result[0]

    res = result.plot() # draw detections on the image as array


    return Image.fromarray(res) # convert array into PIL Image Object
