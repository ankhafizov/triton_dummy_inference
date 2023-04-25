import cv2
import tritonclient.http as httpclient


if __name__ == "__main__":
    img_path = "test_data/car.jpg"
    model_name = "plate_recognition"
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")

    image = cv2.imread(str(img_path))
    print(image.shape)

    inputs, outputs = [], []

    inputs.append(httpclient.InferInput("input__0", image.shape, "UINT8"))
    inputs[0].set_data_from_numpy(image)

    outputs.append(httpclient.InferRequestedOutput(
        "coordinates", binary_data=False))
    outputs.append(httpclient.InferRequestedOutput("texts", binary_data=False))

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
    )

    coordinates = results.as_numpy("coordinates").astype(int)
    texts = results.as_numpy("texts")

    for txt, coord in zip(texts, coordinates):
        x1, y1, x2, y2 = coord
        print(x1, y1, x2, y2)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), [0, 0, 255], 2)
        image = cv2.putText(image, txt, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, [0, 0, 255], 2)

    image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.imshow("plates", image)
    cv2.waitKey(0)
