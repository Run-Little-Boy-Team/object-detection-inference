#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#if __has_include(<lccv.hpp>)
#include <lccv.hpp>
#define LCCV
#endif
#include "net.h"

using namespace std;
using namespace Ort;
using namespace cv;
#ifdef LCCV
using namespace lccv;
#endif

typedef struct
{
    int classId;
    float confidence;
    Rect box;
    int trackId = -1;
    bool liveness = false;
} Result;

class YOLO
{
private:
    string modelPath;
    string inferenceEngine;
    Session *ortModel;
    ncnn::Net ncnnModel;
    YAML::Node configuration;
    float rectConfidenceThreshold;
    float iouThreshold;
    bool verbose;
    int robotId;
    vector<vector<Result>> trackingsList;
    vector<int> trackingCounterList;
    vector<float> preProcessingTimeList;
    vector<float> inferenceTimeList;
    vector<float> postProcessingTimeList;
    vector<float> fpsList;
    ncnn::Mat preProcess(Mat image);
    vector<vector<Result>> postProcess(float *outputs, vector<int> shape, vector<Mat> images);
    void printStats();

public:
    YOLO(string modelPath, string configurationPath, bool gpu, bool verbose);
    ~YOLO();
    vector<vector<Result>> run(vector<Mat> images, bool show);
    vector<vector<Result>> run(vector<string> paths, bool show);
    void tracking(vector<vector<Result>> &resultsList);
    void liveness(vector<vector<Result>> &resultsList, vector<Mat> images);
    vector<Mat> draw(vector<vector<Result>> resultsList, vector<Mat> images, float fps);
    void showDetections(vector<vector<Result>> resultsList, vector<Mat> images, float fps);
    void stream(string video, int webcam, bool show);
};

bool endsWith(string str, string ending);
void printHelp();

#endif