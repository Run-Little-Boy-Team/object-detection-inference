#include "yolo.hpp"

YOLO::YOLO(string modelPath, string configurationPath, bool gpu, bool verbose)
{
    auto t0 = chrono::high_resolution_clock::now();

    cout << fixed << setprecision(2);

    this->configuration = YAML::LoadFile(configurationPath);

    this->modelPath = modelPath;
    if (endsWith(this->modelPath, ".onnx"))
    {
        this->inferenceEngine = "ort";
    }
    else if (endsWith(this->modelPath, ".bin"))
    {
        this->inferenceEngine = "ncnn";
    }
    else
    {
        cout << "Unsupported model format" << endl;
        exit(1);
    }

    if (this->inferenceEngine == "ort")
    {
        static Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");

        SessionOptions sessionOptions;

        if (gpu)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        this->ortModel = new Session(env, this->modelPath.c_str(), sessionOptions);
    }
    else if (this->inferenceEngine == "ncnn")
    {
        if (gpu)
        {
            this->ncnnModel.opt.use_vulkan_compute = true;
        }

        string paramPath = this->modelPath.substr(0, this->modelPath.find_last_of(".")) + ".param";
        this->ncnnModel.load_param(paramPath.c_str());
        this->ncnnModel.load_model(modelPath.c_str());
    }

    this->trackingCounterList = vector<int>();
    this->trackingsList = vector<vector<Result>>();

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
    if (this->inferenceEngine == "ort")
    {
        delete (this->ortModel);
    }
}

vector<vector<Result>> YOLO::run(vector<Mat> images, bool show)
{
    auto t0 = chrono::high_resolution_clock::now();

    int inputSize = this->configuration["input_size"].as<int>();

    vector<ncnn::Mat> preProcessedImages;
    for (Mat image : images)
    {
        ncnn::Mat input = preProcess(image);
        preProcessedImages.push_back(input);
    }

    auto t1 = chrono::high_resolution_clock::now();

    float *outputs;
    vector<int> shape;

    if (this->inferenceEngine == "ort")
    {
        vector<float> input(images.size() * 3 * inputSize * inputSize);
        int size = 3 * inputSize * inputSize;
        for (int i = 0; i < preProcessedImages.size(); i++)
        {
            input.insert(input.begin() + i * size, (float *)preProcessedImages[i].data, (float *)preProcessedImages[i].data + size);
        }

        vector<const char *> inputNodeNames = {"input"};
        vector<const char *> outputNodeNames = {"output"};
        vector<int64_t> inputNodeDims = {(int64_t)images.size(), 3, inputSize, inputSize};
        MemoryInfo memoryInfo = MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Value inputTensor = Value::CreateTensor<float>(memoryInfo, input.data(), input.size(), inputNodeDims.data(), inputNodeDims.size());
        auto outputTensor = ortModel->Run(RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, inputNodeNames.size(), outputNodeNames.data(), outputNodeNames.size());
        auto tensorInfo = outputTensor.front().GetTensorTypeAndShapeInfo();
        vector<int64_t> tensorShape = tensorInfo.GetShape();
        int numElements = tensorInfo.GetElementCount();
        shape = vector<int>(begin(tensorShape), end(tensorShape));
        outputs = outputTensor.front().GetTensorMutableData<float>();
    }
    else if (this->inferenceEngine == "ncnn")
    {
        vector<float> output;
        for (int i = 0; i < preProcessedImages.size(); i++)
        {
            ncnn::Extractor extractor = this->ncnnModel.create_extractor();
            extractor.input("input", preProcessedImages[i]);
            ncnn::Mat ncnnOutput;
            extractor.extract("output", ncnnOutput);
            if (i == 0)
            {
                shape = {(int)images.size(), ncnnOutput.c, ncnnOutput.h, ncnnOutput.w};
            }

            vector<float> vec(ncnnOutput.c * ncnnOutput.h * ncnnOutput.w);
            vec.assign((float *)ncnnOutput.data, (float *)ncnnOutput.data + ncnnOutput.c * ncnnOutput.h * ncnnOutput.w);
            output.insert(output.end(), vec.begin(), vec.end());
        }
        outputs = output.data();
    }

    auto t2 = chrono::high_resolution_clock::now();

    vector<vector<Result>> resultsList = postProcess(outputs, shape, images);

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

ncnn::Mat YOLO::preProcess(Mat image)
{
    int inputSize = this->configuration["input_size"].as<int>();
    ncnn::Mat processed = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, image.cols, image.rows, inputSize, inputSize);
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    processed.substract_mean_normalize(mean_vals, norm_vals);
    return processed;
}

vector<vector<Result>> YOLO::postProcess(float *outputs, vector<int> shape, vector<Mat> images)
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

                    x *= images[b].cols;
                    y *= images[b].rows;
                    w *= images[b].cols;
                    h *= images[b].rows;

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
    this->tracking(resultsList);
    return resultsList;
}

void YOLO::tracking(vector<vector<Result>> &resultsList)
{

    for (int i = 0; i < resultsList.size(); i++)
    {
        if (this->trackingCounterList.size() <= i)
        {
            this->trackingCounterList.push_back(-1);
            this->trackingsList.push_back(vector<Result>());
        }

        for (int j = 0; j < resultsList[i].size(); j++)
        {
            if (this->trackingsList[i].size() <= j)
            {
                resultsList[i][j].trackId = ++this->trackingCounterList[i];
                this->trackingsList[i].push_back(resultsList[i][j]);
            }
            else
            {
                // double minDistance = 1000000;
                // int minIndex = -1;
                // for (int k = 0; k < this->trackingsList[i].size(); k++)
                // {
                //     if (resultsList[i][j].classId != this->trackingsList[i][k].classId)
                //     {
                //         continue;
                //     }
                //     double distance = sqrt(pow(resultsList[i][j].box.x - this->trackingsList[i][k].box.x, 2) + pow(resultsList[i][j].box.y - this->trackingsList[i][k].box.y, 2));
                //     if (distance < minDistance)
                //     {
                //         minDistance = distance;
                //         minIndex = k;
                //     }
                // }
                // if (minDistance < 50)
                // {
                //     resultsList[i][j].trackId = this->trackingsList[i][minIndex].trackId;
                //     this->trackingsList[i][minIndex] = resultsList[i][j];
                // }
                // else
                // {
                //     resultsList[i][j].trackId = ++this->trackingCounterList[i];
                //     this->trackingsList[i].push_back(resultsList[i][j]);
                // }

                double maxIou = 0;
                double maxIndex = -1;
                for (int k = 0; k < this->trackingsList[i].size(); k++)
                {
                    if (resultsList[i][j].classId != this->trackingsList[i][k].classId)
                    {
                        continue;
                    }

                    double x1 = std::max(resultsList[i][j].box.x, trackingsList[i][k].box.x);
                    double y1 = std::max(resultsList[i][j].box.y, trackingsList[i][k].box.y);
                    double x2 = std::min(resultsList[i][j].box.x + resultsList[i][j].box.width, trackingsList[i][k].box.x + trackingsList[i][k].box.width);
                    double y2 = std::min(resultsList[i][j].box.y + resultsList[i][j].box.height, trackingsList[i][k].box.y + trackingsList[i][k].box.height);

                    double intersectionArea = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
                    double unionArea = resultsList[i][j].box.width * resultsList[i][j].box.height + trackingsList[i][k].box.width * trackingsList[i][k].box.height - intersectionArea;

                    double iou = intersectionArea / unionArea;
                    if (iou > maxIou)
                    {
                        maxIou = iou;
                        maxIndex = k;
                    }
                }
                if (maxIou > 0.5)
                {
                    resultsList[i][j].trackId = this->trackingsList[i][maxIndex].trackId;
                    this->trackingsList[i][maxIndex] = resultsList[i][j];
                }
                else
                {
                    resultsList[i][j].trackId = ++this->trackingCounterList[i];
                    this->trackingsList[i].push_back(resultsList[i][j]);
                }

                // double maxSimilarity = 0;
                // double maxIndex = -1;
                // for (int k = 0; k < this->trackingsList[i].size(); k++)
                // {
                //     if (resultsList[i][j].classId != this->trackingsList[i][k].classId)
                //     {
                //         continue;
                //     }

                //     double x1 = std::max(resultsList[i][j].box.x, trackingsList[i][k].box.x);
                //     double y1 = std::max(resultsList[i][j].box.y, trackingsList[i][k].box.y);
                //     double x2 = std::min(resultsList[i][j].box.x + resultsList[i][j].box.width, trackingsList[i][k].box.x + trackingsList[i][k].box.width);
                //     double y2 = std::min(resultsList[i][j].box.y + resultsList[i][j].box.height, trackingsList[i][k].box.y + trackingsList[i][k].box.height);

                //     double intersectionArea = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
                //     double unionArea = resultsList[i][j].box.width * resultsList[i][j].box.height + trackingsList[i][k].box.width * trackingsList[i][k].box.height - intersectionArea;

                //     double iou = intersectionArea / unionArea;

                //     double distance = sqrt(pow(resultsList[i][j].box.x - this->trackingsList[i][k].box.x, 2) + pow(resultsList[i][j].box.y - this->trackingsList[i][k].box.y, 2));

                //     double similarity = iou * (1 / distance);
                //     if (similarity > maxSimilarity)
                //     {
                //         maxSimilarity = similarity;
                //         maxIndex = k;
                //     }
                // }
                // if (maxSimilarity > 0.01)
                // {
                //     resultsList[i][j].trackId = this->trackingsList[i][maxIndex].trackId;
                //     this->trackingsList[i][maxIndex] = resultsList[i][j];
                // }
                // else
                // {
                //     resultsList[i][j].trackId = ++this->trackingCounterList[i];
                //     this->trackingsList[i].push_back(resultsList[i][j]);
                // }
            }
        }
    }
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
            RNG rng(getTickCount());
            Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

            rectangle(image, result.box, color, 3);

            float confidence = floor(100 * result.confidence) / 100;
            string label = this->configuration["classes"].as<vector<string>>()[result.classId] + " " + to_string(result.trackId) + " " + to_string(confidence).substr(0, to_string(confidence).size() - 4);

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

void YOLO::stream(string video, int webcam, bool show)
{
    VideoCapture cap;
#ifdef LCCV
    PiCamera cam;
#endif
    if (video == "pi")
    {
#ifdef LCCV
        cam.startVideo();
#endif
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
#ifdef LCCV
            if (!cam.getVideoFrame(frame, 1000))
            {
                continue;
            }
#endif
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
        vector<vector<Result>> resultsList = run(input, show);

        char c = (char)waitKey(1);
        if (c == 27)
        {
            break;
        }
    }
    if (video == "pi")
    {
#ifdef LCCV
        cam.stopVideo();
#endif
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

bool endsWith(string str, string ending)
{
    if (str.length() >= ending.length())
    {
        return (str.compare(str.length() - ending.length(), ending.length(), ending) == 0);
    }
    else
    {
        return false;
    }
}

void printHelp()
{
    cout << "Usage:" << endl;
    cout << "--model <path-to-your-onnx-model> : Specify the model to use" << endl;
    cout << "--configuration <path-to-your-configuration-file> : Specify the configuration to use" << endl;
    cout << "--gpu : Enable GPU inferences" << endl;
    cout << "--quiet : Disable most of console outputs" << endl;
    cout << "--source <path-to-your-source-file> : Specify a file on which running inferences, could be webcam (camera index, \"pi\" for Pi Camera), image (png, jpg or jpeg) or video (mp4 or avi)" << endl;
    cout << "--hide : Disable showing detections" << endl;
    cout << "--help : print help" << endl;
}