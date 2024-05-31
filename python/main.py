from time import time
import cv2
import numpy as np
from math import floor
import sys
import math
import yaml
import onnxruntime as ort

try:
    import picamera2
except ImportError:
    pass


class Result:
    def __init__(self, class_id: int, confidence: float, box: cv2.typing.Rect) -> None:
        self.class_id = class_id
        self.confidence = confidence
        self.box = box


class YOLO:
    def __init__(
        self,
        model_path: str,
        configuration_path: str,
        gpu: bool,
        verbose: bool = True,
    ) -> None:
        t_0 = time()

        self.configuration = yaml.safe_load(open(configuration_path, "r"))

        session_options = ort.SessionOptions()

        providers = ["CPUExecutionProvider"]
        if gpu:
            providers.insert(0, "CUDAExecutionProvider")
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.model = ort.InferenceSession(model_path, session_options, providers)

        self.rect_confidence_threshold = self.configuration["confidence_threshold"]
        self.iou_threshold = self.configuration["iou_threshold"]
        self.verbose = verbose

        self.pre_processing_time_list = []
        self.inference_time_list = []
        self.post_processing_time_list = []
        self.fps_list = []

        t_1 = time()
        init_time = (t_1 - t_0) * 1000
        if self.verbose:
            print("Initialization:", init_time, "ms")

    def run(
        self, images: list[cv2.typing.MatLike] | list[str], show: bool
    ) -> list[list[Result]]:
        if False not in [isinstance(i, str) for i in images]:
            t_0 = time()

            paths = images

            images = []
            for path in paths:
                image = cv2.imread(path)
                images.append(image)

            t_1 = time()
            reading_images_time = (t_1 - t_0) * 1000
            if self.verbose:
                print("Reading images:", reading_images_time, "ms")

            results_list = self.run(images, False)
            if show:
                self.show_detections(results_list, images, -1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return results_list
        elif False not in [isinstance(i, cv2.typing.MatLike) for i in images]:
            t_0 = time()

            pre_processed_images = []
            for image in images:
                pre_processed_image = self.pre_process(image)
                pre_processed_image = pre_processed_image.transpose((2, 0, 1))
                pre_processed_image = pre_processed_image.astype(np.float32)
                pre_processed_image /= 255.0
                pre_processed_images.append(pre_processed_image)
            pre_processed_images = np.array(pre_processed_images)

            t_1 = time()

            output_tensor = self.model.run(None, {"input": pre_processed_images})
            outputs = output_tensor[0]

            t_2 = time()

            results_list = self.post_process(outputs)

            t_3 = time()

            pre_processing_time = (t_1 - t_0) * 1000
            self.pre_processing_time_list.append(pre_processing_time)
            inference_time = (t_2 - t_1) * 1000
            self.inference_time_list.append(inference_time)
            post_processing_time = (t_3 - t_2) * 1000
            self.post_processing_time_list.append(post_processing_time)
            total_time = (t_3 - t_0) * 1000
            fps = 1 / (total_time / 1000)
            self.fps_list.append(fps)
            if self.verbose:
                print(
                    f"Batch size: {len(pre_processed_images)}\t| Pre-processing: {pre_processing_time:.2f} ms\t| Inference: {inference_time:.2f} ms\t| Post-processing: {post_processing_time:.2f} ms\t| FPS: {fps:.2f}"
                )

            if show:
                if len(images) > 1:
                    self.show_detections(results_list, images, -1)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    self.show_detections(results_list, images, fps)
            return results_list

    def pre_process(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        processed = image

        if processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        else:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        processed = cv2.resize(
            processed,
            (self.configuration["input_size"], self.configuration["input_size"]),
        )
        return processed

    def post_process(self, outputs: cv2.typing.MatLike) -> list[list[Result]]:
        results_list = []

        shape = outputs.shape

        for b in range(shape[0]):
            class_ids = []
            confidences = []
            boxes = []

            for i in range(shape[2]):
                max_class_score = 0
                class_id = 0
                for j in range(shape[3]):
                    for k in range(shape[1] - 5):
                        value = outputs[b][5 + k][i][j]
                        if value >= max_class_score:
                            max_class_score = value
                            class_id = k
                    score = (outputs[b][0][i][j] ** 0.6) * (max_class_score**0.4)
                    if score > self.rect_confidence_threshold:
                        confidences.append(score)
                        class_ids.append(class_id)

                        y = (math.tanh(outputs[b][1][i][j]) + i) / shape[2]
                        x = (math.tanh(outputs[b][2][i][j]) + j) / shape[3]
                        w = 1 / (1 + math.exp(-outputs[b][3][i][j]))
                        h = 1 / (1 + math.exp(-outputs[b][4][i][j]))

                        top_left_x = x - 0.5 * w
                        top_left_y = y - 0.5 * h

                        box = (top_left_x, top_left_y, w, h)
                        boxes.append(box)

            nms_result = cv2.dnn.NMSBoxes(
                boxes, confidences, self.rect_confidence_threshold, self.iou_threshold
            )
            results = []
            for idx in nms_result:
                result = Result(class_ids[idx], confidences[idx], boxes[idx])
                results.append(result)
            results_list.append(results)
        return results_list

    def show_detections(
        self,
        results_list: list[list[Result]],
        images: list[cv2.typing.MatLike],
        fps: float,
    ) -> None:
        t_0 = time()

        for i, results in enumerate(results_list):
            image = images[i]
            for result in results:
                result.box = (
                    int(result.box[0] * image.shape[1]),
                    int(result.box[1] * image.shape[0]),
                    int(result.box[2] * image.shape[1]),
                    int(result.box[3] * image.shape[0]),
                )

                color = np.random.randint(0, 255, 3).tolist()

                image = cv2.rectangle(
                    image,
                    result.box,
                    color,
                    3,
                )

                confidence = floor(100 * result.confidence) / 100
                label = (
                    f"{self.configuration['classes'][result.class_id]} {confidence:.2f}"
                )

                image = cv2.rectangle(
                    image,
                    (result.box[0], result.box[1] - 25),
                    (result.box[0] + len(label) * 15, result.box[1]),
                    color,
                    cv2.FILLED,
                )

                image = cv2.putText(
                    image,
                    label,
                    (result.box[0], result.box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )

            if fps > 0:
                image = cv2.putText(
                    image,
                    f"{fps:.2f} FPS",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            cv2.imshow(str(i), image)

        t_1 = time()
        drawing_time = (t_1 - t_0) * 1000
        if self.verbose:
            print(f"Drawing: {drawing_time:.2f} ms")

    def stream(self, source: int | str, show: bool) -> None:
        if source == "pi":
            camera = picamera2.PiCamera2()
            camera.start_preview()
            preview_configuration = camera.create_preview_configuration()
            camera.configure(preview_configuration)
            camera.start()
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("Error opening video stream or file")
                exit(0)

        while True:
            if source == "pi":
                frame = camera.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            self.run([frame], show)

            c = cv2.waitKey(1)
            if c == 27:
                break
        if source == "pi":
            camera.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        self.print_stats()

    def print_stats(self) -> None:
        average_pre_processing_time = sum(self.pre_processing_time_list) / len(
            self.pre_processing_time_list
        )
        average_inference_time = sum(self.inference_time_list) / len(
            self.inference_time_list
        )
        average_post_processing_time = sum(self.post_processing_time_list) / len(
            self.post_processing_time_list
        )
        average_fps = sum(self.fps_list) / len(self.fps_list)
        print(
            f"Average times:\nPre-processing: {average_pre_processing_time:.2f} ms\t| Inference: {average_inference_time:.2f} ms\t| Post-processing: {average_post_processing_time:.2f} ms\t| FPS: {average_fps:.2f}"
        )


def print_help() -> None:
    print("Usage:")
    print("--model <path-to-your-onnx-model> : Specify the model to use")
    print(
        "--configuration <path-to-your-configuration-file> : Specify the configuration to use"
    )
    print("--gpu : Enable GPU inferences")
    print("--quiet : Disable most of console outputs")
    print(
        "--source <path-to-your-source-file> : Specify a file on which running"
        ' inferences, could be webcam (camera index, "pi" for Pi Camera), image (png, jpg or jpeg) or'
        " video (mp4 or avi)"
    )
    print("--hide : Disable showing detections")
    print("--help : print help")


if __name__ == "__main__":
    model_path = "./models/model.onnx"
    configuration_path = "./config.yaml"
    gpu = False
    verbose = True
    source = 0
    show = True
    args = sys.argv
    if len(args) < 2:
        print_help()
        exit(0)
    skip = False
    for i, arg in enumerate(args):
        if skip:
            skip = False
            continue
        if arg == "--model":
            model_path = args[i + 1]
            skip = True
        elif arg == "--configuration":
            configuration_path = args[i + 1]
            skip = True
        elif arg == "--gpu":
            gpu = True
        elif arg == "--quiet":
            verbose = False
        elif arg == "--source":
            source = args[i + 1]
            skip = True
        elif arg == "--hide":
            show = False
        elif args[i] == "--help":
            print_help()
            exit(0)
    yolo: YOLO = YOLO(model_path, configuration_path, gpu, verbose)
    if source == "pi":
        yolo.stream(source, show)
    else:
        try:
            webcam = int(source)
            yolo.stream(webcam, show)
        except ValueError:
            videos = [".mp4", ".avi", ".webm"]
            for video in videos:
                if source.endswith(video):
                    yolo.stream(source, show)
                    exit(0)
            images = [".jpg", ".jpeg", ".png"]
            for image in images:
                if source.endswith(image):
                    yolo.run([source], show)
                    exit(0)
