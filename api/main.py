import io
import json
import logging
import os
import sys
import time

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils import Results, resnet_inference, generate_image_hash

MODEL_NAME = os.environ.get("MODEL_NAME")
POSTGRES_CONN_STRING = os.environ.get("POSTGRES_CONN_STRING")
TRITON_URL = os.environ.get("TRITON_URL", "triton-inference-server")
TRITON_PORT = os.environ.get("TRITON_PORT", "8000")
log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
LOG_LEVEL = log_levels.get(os.environ.get("LOG_LEVEL", "info").lower())
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
triton_client = httpclient.InferenceServerClient(TRITON_URL, concurrency=10)
logging.info(f"{triton_client.get_model_repository_index() = }")

with open("imagenet_classes.txt", "r") as f:
    classes = [" ".join(x.strip().split(" ")[1:]) for x in f.readlines()]

# Create the SQLAlchemy engine and session
engine = create_engine(POSTGRES_CONN_STRING)
Session = sessionmaker(bind=engine)
Results.metadata.create_all(engine)


@app.get("/clf-app/")
async def classify(request: Request):
    logging.debug(f"{request.headers.get('x-real-ip') = }")
    logging.debug(f"{request.client.host = }")
    return FileResponse("static/index.html")


@app.post("/clf-app/uploadfile/")
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    logging.debug(f"{request.headers = }")
    client_ip = request.headers.get("x-real-ip") or request.client.host
    image = Image.open(io.BytesIO(await file.read()))
    img_hash = generate_image_hash(image)
    image = np.array(image)
    t0 = time.time()
    results = resnet_inference(image, triton_client, MODEL_NAME, classes)
    inf_duration = time.time() - t0
    response_data = list(results.values())[0]

    db_result = Results(
        user_ip=client_ip,
        image_hash=img_hash,
        inference_duration_sec=inf_duration,
        predicted_class=response_data["class_name"],
        probability=response_data["probability"],
    )
    session = Session()
    session.add(db_result)
    session.commit()
    session.close()
    return Response(json.dumps(response_data), media_type="application/json")
