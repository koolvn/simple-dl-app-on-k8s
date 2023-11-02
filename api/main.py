import io
import json
import logging
import sys

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from utils import resnet_inference

MODEL_NAME = "resnet18"
LOG_LEVEL = logging.DEBUG
root = logging.getLogger()
root.setLevel(LOG_LEVEL)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(LOG_LEVEL)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)

logging.info("Starting API")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.info("Starting triton client")
triton_client = httpclient.InferenceServerClient("triton-inference-server:8000")
logging.info(f"{triton_client.get_model_repository_index() = }")

with open("imagenet_classes.txt", "r") as f:
    classes = [" ".join(x.strip().split(" ")[1:]) for x in f.readlines()]


@app.get("/clf-app/")
async def classify():
    return FileResponse("static/index.html")


@app.post("/clf-app/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)
    results = resnet_inference(image, triton_client, MODEL_NAME, classes)
    response_data = list(results.values())[0]
    return Response(json.dumps(response_data), media_type="application/json")
