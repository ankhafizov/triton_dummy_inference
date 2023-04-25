from __future__ import annotations

import cv2
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput


def send_request(
    client, model_name: str, input_shape: tuple[int, ...], input_type: str
) -> np.ndarray:
    inputs, outputs = [], []

    inputs.append(InferInput("input__0", input_shape, input_type))
    input_dtype = np.float32 if input_type == "FP32" else np.uint8
    inputs[0].set_data_from_numpy(
        np.random.randn(*input_shape).astype(input_dtype))

    outputs.append(InferRequestedOutput(
        "output__0", binary_data=False))
    results = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )
    return results.as_numpy("output__0")


if __name__ == "__main__":
    model_name = "yolo"
    input_shape = (500, 800, 3)
    output_shape = (0,)  # no detections
    input_type = "UINT8"
    triton_client = InferenceServerClient(url="localhost:8000")

    prediction = send_request(
        triton_client, model_name, input_shape, input_type)
    print(prediction)
