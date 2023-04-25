import cv2
import tritonclient.http as httpclient
import time
import numpy as np


class FPS_Counter:
    def __init__(self, calc_time_perion_N_frames: int) -> None:
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_perion_N_frames

    def calc_FPS(self) -> float:
        time_buffer_is_full = len(
            self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / \
                (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0


if __name__ == "__main__":
    video_path = "test_data/video.mp4"
    model_name = "plate_recognition"
    fps_N_frames = 10

    triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    cap = cv2.VideoCapture(video_path)
    fps_counter = FPS_Counter(fps_N_frames)

    while True:
        ret, image = cap.read()
        inputs, outputs = [], []
        inputs.append(httpclient.InferInput("input__0", image.shape, "UINT8"))
        inputs[0].set_data_from_numpy(image)

        outputs.append(httpclient.InferRequestedOutput(
            "coordinates", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput(
            "texts", binary_data=False))

        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
        )

        coordinates = results.as_numpy("coordinates").astype(int)
        texts = results.as_numpy("texts")

        for txt, coord in zip(texts, coordinates):
            x1, y1, x2, y2 = coord
            image = cv2.resize(image, (1920, 1080),
                               interpolation=cv2.INTER_AREA)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), [0, 0, 255], 2)
            image = cv2.putText(image, txt, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, [0, 0, 255], 2)

        image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.imshow("plates", image)
        cv2.waitKey(10)

        fps = fps_counter.calc_FPS()
        if fps > 0:
            print(f"fps({fps_N_frames} frames): {fps}")
