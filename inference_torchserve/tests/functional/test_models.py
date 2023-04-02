import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import requests

from inference_torchserve.data_models import PlatePrediction


@pytest.mark.parametrize(
    "image,expected_plates,expected_shape",
    [
        (
            Path("tests/data/car.jpg"),
            [
                PlatePrediction(
                    xmin=232, ymin=813, xmax=324, ymax=842, confidence=0.929
                ),
                PlatePrediction(
                    xmin=1097, ymin=661, xmax=1142, ymax=674, confidence=0.892
                ),
                PlatePrediction(
                    xmin=1521, ymin=640, xmax=1566, ymax=652, confidence=0.861
                ),
                PlatePrediction(
                    xmin=1286, ymin=635, xmax=1316, ymax=644, confidence=0.689
                ),
            ],
            (4, 3, 24, 94),
        ),
        (Path("tests/data/cat.jpeg"), [], (0,)),
    ],
)
def test_yolo(
    image: Path, expected_plates: List[PlatePrediction], expected_shape: Tuple[int]
):
    response = requests.post(
        "http://localhost:8080/predictions/yolo", data=image.open("rb").read()
    )
    prediction = response.json()
    for pred_plate, exp_plate in zip(prediction["coordinates"], expected_plates):
        pred_plate = PlatePrediction.parse_obj(pred_plate)
        assert pred_plate.xmin == exp_plate.xmin
        assert pred_plate.ymin == exp_plate.ymin
        assert pred_plate.xmax == exp_plate.xmax
        assert pred_plate.ymax == exp_plate.ymax
        assert pred_plate.confidence == pytest.approx(exp_plate.confidence, abs=1e-3)

    assert np.array(prediction["data"]).shape == expected_shape


def test_stn():
    input_shape = (4, 3, 24, 94)
    output_shape = (4, 3, 24, 94)

    inputs = {"data": np.random.randn(*input_shape).astype(np.float32).tolist()}
    inputs = json.dumps(inputs).encode("utf-8")

    response = requests.post("http://localhost:8080/predictions/stn", data=inputs)

    output = np.array(response.json()["data"])

    assert output.shape == output_shape


def test_lprnet():
    input_shape = (4, 3, 24, 94)

    inputs = {"data": np.random.randn(*input_shape).astype(np.float32).tolist()}
    inputs = json.dumps(inputs).encode("utf-8")

    response = requests.post("http://localhost:8080/predictions/lprnet", data=inputs)

    texts = response.json()["data"]
    for text in texts:
        assert isinstance(text, str)


@pytest.mark.parametrize(
    "img_path,expected_coordinates,expected_texts",
    [
        [
            "tests/data/car.jpg",
            np.array(
                [
                    [232, 813, 324, 842],
                    [1097, 661, 1142, 674],
                    [1521, 640, 1566, 652],
                    [1286, 635, 1316, 644],
                ]
            ),
            ["B840OK197", "", "", ""],
        ],
        ["tests/data/cat.jpeg", [], []],
    ],
)
def test_plate_recognition(
    img_path: Path,
    expected_coordinates: List[List[int]],
    expected_texts: List[str],
):
    image = Path(img_path).open("rb").read()
    response = requests.post(
        "http://localhost:8080/wfpredict/plate_recognition",
        data=image,
    )

    texts = response.json()["texts"]
    coordinates = response.json()["coordinates"]

    assert texts == expected_texts

    for act_coord, exp_coord in zip(coordinates, expected_coordinates):
        assert act_coord["xmin"] == exp_coord[0]
        assert act_coord["ymin"] == exp_coord[1]
        assert act_coord["xmax"] == exp_coord[2]
        assert act_coord["ymax"] == exp_coord[3]