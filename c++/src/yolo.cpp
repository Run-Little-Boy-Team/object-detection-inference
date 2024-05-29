#include "yolo.hpp"

YOLO::YOLO(string modelPath, string configurationPath, bool gpu, bool verbose)
{
    auto t0 = chrono::high_resolution_clock::now();

    cout << fixed << setprecision(2);

    this->configuration = YAML::LoadFile(configurationPath);

    static Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");

    SessionOptions sessionOptions;

    if (gpu)
    {
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    this->model = new Session(env, modelPath.c_str(), sessionOptions);

    this->rectConfidenceThreshold = this->configuration["confidence_threshold"].as<float>();
    this->iouThreshold = this->configuration["iou_threshold"].as<float>();
    this->verbose = verbose;

    auto t1 = chrono::high_resolution_clock::now();
    float initTime = chrono::duration<float, milli>(t1 - t0).count();
    if (this->verbose)
    {
        cout << "Initialization: " << initTime << " ms" << endl;
    }
}

YOLO::~YOLO()
{
    delete (this->model);
}

vector<vector<Result>> YOLO::run(vector<Mat> images, bool show)
{
    auto t0 = chrono::high_resolution_clock::now();

    int inputSize = this->configuration["input_size"].as<int>();
    vector<float> preProcessedImages(images.size() * 3 * inputSize * inputSize);
    for (int i = 0; i < images.size(); i++)
    {
        Mat preProcessedImage = preProcess(images[i]);
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < inputSize; h++)
            {
                for (int w = 0; w < inputSize; w++)
                {
                    int index = i * 3 * inputSize * inputSize + c * inputSize * inputSize + h * inputSize + w;
                    preProcessedImages[index] = static_cast<float>(preProcessedImage.at<Vec3b>(h, w)[c]) / 255.0f;
                }
            }
        }
    }

    auto t1 = chrono::high_resolution_clock::now();

    float *outputs;
    vector<int> shape;

    vector<const char *> inputNodeNames = {"input"};
    vector<const char *> outputNodeNames = {"output"};
    vector<int64_t> inputNodeDims = {(int64_t)images.size(), 3, inputSize, inputSize};
    MemoryInfo memoryInfo = MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Value inputTensor = Value::CreateTensor<float>(memoryInfo, preProcessedImages.data(), preProcessedImages.size(), inputNodeDims.data(), inputNodeDims.size());
    auto outputTensor = model->Run(RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, inputNodeNames.size(), outputNodeNames.data(), outputNodeNames.size());
    auto tensorInfo = outputTensor.front().GetTensorTypeAndShapeInfo();
    vector<int64_t> tensorShape = tensorInfo.GetShape();
    int numElements = tensorInfo.GetElementCount();
    shape = vector<int>(begin(tensorShape), end(tensorShape));
    outputs = outputTensor.front().GetTensorMutableData<float>();

    auto t2 = chrono::high_resolution_clock::now();

    vector<vector<Result>> resultsList = postProcess(outputs, shape);

    auto t3 = chrono::high_resolution_clock::now();
    float preProcessingTime = chrono::duration<float, milli>(t1 - t0).count();
    this->preProcessingTimeList.push_back(preProcessingTime);
    float inferenceTime = chrono::duration<float, milli>(t2 - t1).count();
    this->inferenceTimeList.push_back(inferenceTime);
    float postProcessingTime = chrono::duration<float, milli>(t3 - t2).count();
    this->postProcessingTimeList.push_back(postProcessingTime);
    float totalTime = chrono::duration<float, milli>(t3 - t0).count();
    float fps = 1 / (totalTime / 1000);
    this->fpsList.push_back(fps);
    if (this->verbose)
    {
        cout << "Batch size: " << images.size() << "\t| Pre-processing: " << preProcessingTime << " ms\t| Inference: " << inferenceTime << " ms\t| Post-processing: " << postProcessingTime << " ms\t| FPS: " << fps << endl;
    }

    if (show)
    {
        if (images.size() > 1)
        {
            showDetections(resultsList, images, -1);
            waitKey(0);
            destroyAllWindows();
        }
        else
        {
            showDetections(resultsList, images, fps);
        }
    }
    return resultsList;
}

vector<vector<Result>> YOLO::run(vector<string> paths, bool show)
{
    auto t0 = chrono::high_resolution_clock::now();

    vector<Mat> images;
    for (string path : paths)
    {
        Mat image = imread(path);
        images.push_back(image);
    }

    auto t1 = chrono::high_resolution_clock::now();
    float readingImagesTime = chrono::duration<float, milli>(t1 - t0).count();
    if (this->verbose)
    {
        cout << "Reading images: " << readingImagesTime << " ms" << endl;
    }

    vector<vector<Result>> resultsList = run(images, false);
    if (show)
    {
        showDetections(resultsList, images, -1);
        waitKey(0);
        destroyAllWindows();
    }
    return resultsList;
}

Mat YOLO::preProcess(Mat image)
{
    Mat processed = image.clone();
    if (processed.channels() == 3)
    {
        cvtColor(processed, processed, COLOR_BGR2RGB);
    }
    else
    {
        cvtColor(processed, processed, COLOR_GRAY2RGB);
    }

    int inputSize = this->configuration["input_size"].as<int>();
    resize(processed, processed, Size(inputSize, inputSize));
    return processed;
}

vector<vector<Result>> YOLO::postProcess(float *outputs, vector<int> shape)
{
    vector<vector<Result>> resultsList;

    for (int b = 0; b < shape[0]; b++)
    {
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (int i = 0; i < shape[2]; i++)
        {
            float maxClassScore = 0;
            int classId = 0;
            for (int j = 0; j < shape[2]; j++)
            {
                for (int k = 0; k < shape[1] - 5; k++)
                {
                    float value = outputs[(b * shape[1] * shape[2] * shape[3]) + ((5 + k) * shape[2] * shape[3]) + (i * shape[3]) + j];
                    if (value >= maxClassScore)
                    {
                        maxClassScore = value;
                        classId = k;
                    }
                }

                float score = pow(outputs[(b * shape[1] * shape[2] * shape[3]) + (0 * shape[2] * shape[3]) + (i * shape[3]) + j], 0.6) * pow(maxClassScore, 0.4);
                if (score > this->rectConfidenceThreshold)
                {
                    confidences.push_back(maxClassScore);
                    classIds.push_back(classId);

                    float y = (tanh(outputs[(b * shape[1] * shape[2] * shape[3]) + (1 * shape[2] * shape[3]) + (i * shape[3]) + j]) + i) / shape[2];
                    float x = (tanh(outputs[(b * shape[1] * shape[2] * shape[3]) + (2 * shape[2] * shape[3]) + (i * shape[3]) + j]) + j) / shape[3];
                    float w = 1 / (1 + exp(-outputs[(b * shape[1] * shape[2] * shape[3]) + (3 * shape[2] * shape[3]) + (i * shape[3]) + j]));
                    float h = 1 / (1 + exp(-outputs[(b * shape[1] * shape[2] * shape[3]) + (4 * shape[2] * shape[3]) + (i * shape[3]) + j]));

                    x *= 100;
                    y *= 100;
                    w *= 100;
                    h *= 100;

                    float top_left_x = x - 0.5 * w;
                    float top_left_y = y - 0.5 * h;

                    Rect box(top_left_x, top_left_y, w, h);
                    boxes.push_back(box);
                }
            }
        }
        vector<int> nmsResult;
        dnn::NMSBoxes(boxes, confidences, this->rectConfidenceThreshold, this->iouThreshold, nmsResult);
        vector<Result> results;
        for (int idx : nmsResult)
        {
            Result result;
            result.classId = classIds[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            results.push_back(result);
        }
        resultsList.push_back(results);
    }
    return resultsList;
}

void YOLO::showDetections(vector<vector<Result>> resultsList, vector<Mat> images, float fps)
{
    auto t0 = chrono::high_resolution_clock::now();

    for (int i = 0; i < resultsList.size(); i++)
    {
        Mat image = images[i].clone();
        for (int j = 0; j < resultsList[i].size(); j++)
        {
            Result result = resultsList[i][j];
            result.box.x = (int)(result.box.x * image.cols / 100);
            result.box.y = (int)(result.box.y * image.rows / 100);
            result.box.width = (int)(result.box.width * image.cols / 100);
            result.box.height = (int)(result.box.height * image.rows / 100);
            RNG rng(getTickCount());
            Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

            rectangle(image, result.box, color, 3);

            float confidence = floor(100 * result.confidence) / 100;
            string label = this->configuration["classes"].as<vector<string>>()[result.classId] + " " +
                           to_string(confidence).substr(0, to_string(confidence).size() - 4);

            rectangle(
                image,
                Point(result.box.x, result.box.y - 25),
                Point(result.box.x + label.length() * 15, result.box.y),
                color,
                FILLED);

            putText(
                image,
                label,
                Point(result.box.x, result.box.y - 5),
                FONT_HERSHEY_SIMPLEX,
                0.75,
                Scalar(0, 0, 0),
                2);
        }
        if (fps > 0)
        {
            stringstream stream;
            stream << fixed << setprecision(2) << fps;
            string s = stream.str();
            putText(
                image,
                s + " FPS",
                Point(20, 40),
                FONT_HERSHEY_SIMPLEX,
                1,
                Scalar(0, 255, 0),
                2);
        }
        imshow(to_string(i), image);
    }

    auto t1 = chrono::high_resolution_clock::now();
    float drawingTime = chrono::duration<float, milli>(t1 - t0).count();
    if (this->verbose)
    {
        cout << "Drawing: " << drawingTime << " ms" << endl;
    }
}

void YOLO::stream(string video, int webcam)
{
    VideoCapture cap;
    PiCamera camera;
    if (video == "pi")
    {
        cam.startVideo();
    }
    else
    {
        if (video != "")
        {
            cap = VideoCapture(video);
        }
        else if (webcam >= 0)
        {
            cap = VideoCapture(webcam);
        }
        else
        {
            cout << "No stream source selected" << endl;
        }

        if (!cap.isOpened())
        {
            cout << "Error opening video stream or file" << endl;
            return;
        }
    }

    while (true)
    {
        Mat frame;
        if (video == "pi")
        {
            if (!cam.getVideoFrame(frame, 1000))
            {
                continue;
            }
        }
        else
        {
            cap >> frame;
        }

        if (frame.empty())
        {
            break;
        }

        vector<Mat> input;
        input.push_back(frame);
        vector<vector<Result>> resultsList = run(input, true);

        char c = (char)waitKey(1);
        if (c == 27)
        {
            break;
        }
    }
    if (video == "pi")
    {
        cam.stopVideo();
    }
    else
    {
        cap.release();
    }
    destroyAllWindows();
    printStats();
}

void YOLO::printStats()
{
    float averagePreProcessingTime = 0;
    float averageInferenceTime = 0;
    float averagePostProcessingTime = 0;
    float averageFps = 0;
    for (int i = 0; i < this->fpsList.size(); i++)
    {
        averagePreProcessingTime += this->preProcessingTimeList[i];
        averageInferenceTime += this->inferenceTimeList[i];
        averagePostProcessingTime += this->postProcessingTimeList[i];
        averageFps += this->fpsList[i];
    }
    averagePreProcessingTime /= this->preProcessingTimeList.size();
    averageInferenceTime /= this->inferenceTimeList.size();
    averagePostProcessingTime /= this->postProcessingTimeList.size();
    averageFps /= this->fpsList.size();
    cout
        << "Average times:\nPre-processing: " << averagePreProcessingTime << " ms\t| Inference: " << averageInferenceTime << " ms\t| Post-processing: " << averagePostProcessingTime << " ms\t| FPS: " << averageFps << endl;
}
