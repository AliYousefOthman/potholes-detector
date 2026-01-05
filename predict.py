from ultralytics import YOLO
from PIL import Image
import uvicorn


def predict_img(img,model):
    result = model.predict(img)

    result = result[0]

    res = result.plot()

    return Image.fromarray(res)