import io
import json
import logging
import sys

import cv2
import numpy as np
import tritonclient
import tritonclient.http as httpclient
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

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


def resnet_preprocess(
    img: np.ndarray, new_size=(224, 224), swapRB=False, dtype=np.float32
):
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]
    _img: np.ndarray = (img / 255.0 - _mean) / _std
    if swapRB:
        _img = _img[..., ::-1]
    _img = cv2.resize(_img, new_size)
    _img = _img.transpose(2, 0, 1)
    return _img[None, ...].astype(dtype)


def resnet_inference(img: np.ndarray, client: tritonclient.http, model_name: str):
    cfg = client.get_model_config(model_name)

    _input_shape = cfg["input"][0]["dims"]
    _input_name = cfg["input"][0]["name"]
    _input_dtype = cfg["input"][0]["data_type"].split("_")[-1]
    _output_name = cfg["output"][0]["name"]

    tensor = resnet_preprocess(img, new_size=_input_shape[1:])

    _input = httpclient.InferInput(_input_name, tensor.shape, datatype=_input_dtype)
    _input.set_data_from_numpy(tensor, binary_data=True)
    _response = client.infer(model_name=MODEL_NAME, inputs=[_input])

    results = _response.as_numpy(_output_name)
    predicted_class = np.argmax(results)

    proba = results.squeeze()[predicted_class]
    class_name = classes[predicted_class]
    return class_name, proba


@app.get("/")
async def classify():
    return FileResponse("static/index.html")


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)
    cls_name, proba = resnet_inference(image, triton_client, MODEL_NAME)
    response_data = {
        "class_name": cls_name,
        "probability": f"{proba:.1f}"
    }
    return Response(json.dumps(response_data), media_type="application/json")
