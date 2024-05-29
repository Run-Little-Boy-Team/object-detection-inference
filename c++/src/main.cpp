#include "yolo.hpp"

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
    cout << "--help : print help" << endl;
}

int main(int argc, char *argv[])
{
    string modelPath = "./models/model.onnx";
    string configurationPath = "./config.yaml";
    bool gpu = false;
    bool verbose = true;
    string source = "0";
    if (argc < 2)
    {
        printHelp();
        exit(0);
    }
    for (int i = 0; i < argc; i++)
    {
        if (strcmp(argv[i], "--model") == 0)
        {
            modelPath = argv[++i];
        }
        else if (strcmp(argv[i], "--configuration") == 0)
        {
            configurationPath = argv[++i];
        }
        else if (strcmp(argv[i], "--gpu") == 0)
        {
            gpu = true;
        }
        else if (strcmp(argv[i], "--quiet") == 0)
        {
            verbose = false;
        }
        else if (strcmp(argv[i], "--source") == 0)
        {
            source = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            printHelp();
            exit(0);
        }
    }
    YOLO yolo(modelPath, configurationPath, gpu, verbose);
    if (source == "pi")
    {
        yolo.stream(source, -1);
    }
    else
    {
        try
        {
            int webcam = stoi(source);
            yolo.stream("", webcam);
        }
        catch (const invalid_argument &e)
        {
            vector<string> videos = {".mp4", ".avi"};
            for (string video : videos)
            {
                if (endsWith(source, video))
                {
                    yolo.stream(source, -1);
                    return 0;
                }
            }
            vector<string> images = {".jpg", ".jpeg", ".png"};
            for (string image : images)
            {
                if (endsWith(source, image))
                {
                    vector<string> input = {source};
                    yolo.run(vector<string>(input), true);
                    return 0;
                }
            }
        }
    }

    return 0;
}
