from typing import Any

import cv2
import numpy as np
import tritonclient
from tritonclient.http import InferInput


def parse_model_metadata(
    meta: dict,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    for idx, inp in enumerate(meta["inputs"]):
        _name = inp["name"]
        _shape = inp["shape"]
        _dtype = inp["datatype"]
        inputs.append({"name": _name, "shape": _shape, "datatype": _dtype})

    outputs = []
    for idx, out in enumerate(meta["outputs"]):
        _name = out["name"]
        _shape = out["shape"]
        _dtype = out["datatype"]
        outputs.append({"name": _name, "shape": _shape, "datatype": _dtype})
    return inputs, outputs


def resnet_preprocess(
    img: np.ndarray, new_size=(224, 224), swapRB=True, dtype=np.float32
) -> np.ndarray:
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]
    _img: np.ndarray = (img / 255.0 - _mean) / _std
    if swapRB:
        _img = _img[..., ::-1]
    _img = cv2.resize(_img, new_size)
    _img = _img.transpose(2, 0, 1)
    return _img[None, ...].astype(dtype)


def resnet_inference(
    img: np.ndarray, client: tritonclient.http, model_name: str, classes: list
) -> dict:
    inputs, outputs = parse_model_metadata(client.get_model_metadata(model_name))
    inputs_list = []
    for inp in inputs:
        # TODO: find a smarter way to get rid of batch dim
        new_size = inp["shape"][2:] if len(inp["shape"]) == 4 else inp["shape"][1:]
        tensor = resnet_preprocess(img, new_size=new_size)
        inp["shape"] = tensor.shape
        _input = InferInput(**inp)
        _input.set_data_from_numpy(tensor, binary_data=True)
        inputs_list.append(_input)

    _response = client.infer(model_name=model_name, inputs=inputs_list)

    results = {}
    for out in outputs:
        _res = _response.as_numpy(out["name"])
        predicted_class = np.argmax(_res)
        proba = _res.squeeze()[predicted_class]
        class_name = classes[predicted_class]
        results[out["name"]] = {"class_name": class_name, "probability": f"{proba:.1f}"}

    return results
