from fastapi import FastAPI,File,UploadFile,status
from fastapi.responses import StreamingResponse
from io import BytesIO
from ultralytics import YOLO
from PIL import Image
from predict import predict_img

model1 = YOLO(r'runs\detect\yolov8n.pt\weights\best.pt')

model2 = YOLO(r'runs\detect\yolov8s.pt\weights\best.pt')

app = FastAPI()

@app.post("/api/model1")
async def upload(file : UploadFile = File(...)):

    image = Image.open(file.file) # image uploaded
    predicted_image = predict_img(image,model1)

    buffer = BytesIO() # in memory bytes that acts like a file
    predicted_image.save(buffer,format='PNG')
    buffer.seek(0)
    return StreamingResponse(buffer,status_code=status.HTTP_202_ACCEPTED,media_type='image/png')


@app.post("/api/model2")
async def upload(file : UploadFile = File(...)):

    image = Image.open(file.file) # image uploaded
    predicted_image = predict_img(image,model2)

    buffer = BytesIO() # in memory bytes that acts like a file
    predicted_image.save(buffer,format='PNG') # saved in ram (acts like file)
    buffer.seek(0)
    return StreamingResponse(buffer,status_code=status.HTTP_202_ACCEPTED,media_type='image/png')
