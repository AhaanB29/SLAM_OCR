// MIT License

// Copyright (c) 2024 A2va

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <numeric>
#include <array>
#include <tuple>
#include <cmath>

#include <xtensor.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <onnxruntime_cxx_api.h>

template <typename T> 
T custom_mean(const xt::xarray<T> &x){ 
    return std::pow(xt::prod(x)(), 2.0 / std::sqrt(x.size()));
}

cv::Mat normalize(const cv::Mat &img) {
    cv::Mat fimg;
    // Convert to float
    img.convertTo(fimg, CV_32F);

    cv::Mat dst;
    cv::normalize(fimg, dst, -1, 1, cv::NORM_MINMAX);
    return dst;
}

cv::Mat normalizeAndPAD(const cv::Mat &img, cv::Size maxSize) { 
    cv::Mat n = normalize(img);
    cv::Mat padImg = cv::Mat::zeros(maxSize, CV_32F);

    // Right pad
    const cv::Rect roi = cv::Rect(0, 0, img.cols, img.rows);
    n.copyTo(padImg(roi));

    if (maxSize.width != img.cols)
    {
        // Border pad
        cv::copyMakeBorder(n, padImg, 0, 0, img.cols, maxSize.width - img.cols, cv::BORDER_REPLICATE);
    }
    return padImg;
}

std::vector<Ort::AllocatedStringPtr> getNodeNames(Ort::Session *session, bool getInputNames) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t numNodes = getInputNames ? session->GetInputCount() : session->GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> nodeNamesPtr;
    nodeNamesPtr.reserve(numNodes);

    // Iterate over all input/output nodes
    for (size_t i = 0; i < numNodes; i++) {
        auto nodeName = getInputNames ? session->GetInputNameAllocated(i, allocator)
                                      : session->GetOutputNameAllocated(i, allocator);
        nodeNamesPtr.push_back(std::move(nodeName));
    }
    return nodeNamesPtr;
}


template<typename T>
std::tuple<xt::xarray<T>, xt::xarray<T>> max(const xt::xarray<T> &array, std::size_t axis, bool keepdims) {
    const auto max_ = xt::amax(array, axis);
    const auto argmax = xt::argmax(array, axis);

    if(keepdims) {
        return std::make_tuple(xt::expand_dims(max_, axis), xt::expand_dims(argmax, axis));
    }
    return std::make_tuple(max_, argmax);
}


template<typename T>
xt::xarray<T> softmax(const xt::xarray<T>& t, std::size_t axis = 0) {
    const auto [m, _] = max(t, axis, true);
    const auto sub = t - xt::broadcast(m, t.shape());
    const auto exp = xt::exp(sub);

    const auto sum = xt::sum(exp, axis);

    return exp / xt::broadcast(xt::expand_dims(sum, axis), t.shape());
}

template<typename T>
xt::xarray<bool> not_repeated(const xt::xarray<T>& t) {
    // Create a boolean array where true is when the value is not repeated
    xt::xarray<bool> a = xt::not_equal(xt::view(t, xt::range(1, xt::placeholders::_)), xt::view(t, xt::range(0, -1)));
    // Insert a 'true' at the beginning of the array
    xt::xarray<bool> result = xt::empty<bool>(t.shape());
    result(0) = true;

    xt::view(result, xt::range(1, xt::placeholders::_)) = a;
    return result;
}

std::vector<std::string> readCharactersFile(std::filesystem::path p) {
    if (!std::filesystem::exists(p)) {
        return std::vector<std::string>();
    }

    std::vector<std::string> v;
    v.push_back("[blank]");
    
    std::ifstream file(p);
    std::string line;
    while (std::getline(file, line)) {
        v.push_back(line);
    }

    return v;
}

template<typename T>
std::tuple<std::string, xt::xarray<int>> greedyDecode(const xt::xarray<T> &indexs, std::vector<std::string> characters) {

    xt::xarray<bool> a = not_repeated(indexs);

    std::vector<int> ignore_idx = {0};  // Ignore blank character
    xt::xarray<bool> b = !(xt::isin(indexs, xt::adapt(ignore_idx)));

    xt::xarray<bool> c = a & b;

    const auto indices = xt::nonzero(c);
    xt::xarray<int> indexed_text = xt::index_view(indexs, xt::flatten_indices(indices));

    std::string text;
    for (const auto i : indexed_text) {
        text.append(characters[i]);
    }
    return std::make_tuple(text, indexed_text);
}

std::tuple<std::string, float> inference(Ort::Session *session, cv::Mat img, std::filesystem::path charactersPath)
{
    // Preprocess the image
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);


    int max_width = img.size().width;
    std::vector<int64_t> modelInputShape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // If the onnx model doensn't have dynamic inputs
    if (modelInputShape[3] != -1) {
        max_width = modelInputShape.back();
    }
    int imgH = modelInputShape[2];

    cv::Mat modelInput;
    const cv::Size size = cv::Size(max_width, imgH);
    cv::resize(img, modelInput, size);
    modelInput = normalizeAndPAD(modelInput, size);

    // Get model input/ouput names
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr = getNodeNames(session, true);
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr = getNodeNames(session, false);
    const std::vector<const char *> inputNames = {inputNamesPtr.data()->get()};
    const std::vector<const char *> outputNames = {outputNamesPtr.data()->get()};

     // Create input tensor
    std::array<int64_t, 4> inputShape{1, 1, modelInput.rows, modelInput.cols};

    const auto inputTensorSize = modelInput.cols * modelInput.rows * modelInput.channels();
    const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, (float *)modelInput.data, modelInput.size().area(), inputShape.data(), inputShape.size());

    try
    {
        // Inference
        auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, inputNames.size(),
                                         outputNames.data(), outputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());

        // Remainder floatArray pointer is freed as soon as outputTensor goes out of scope
        float *floatArray = outputTensor.front().GetTensorMutableData<float>();

        std::array<std::size_t, 2> shape = {outputShape[1], outputShape[2]};
        xt::xarray<float> preds = xt::adapt(floatArray, outputCount, xt::no_ownership(), shape);

        const auto predsProbs = softmax(preds, 1);
        const auto norm = xt::sum(predsProbs, 1);
        xt::xarray<float> probs = predsProbs / xt::expand_dims(norm, 1);

        const auto [maxs, indexs] = max(probs, 1, true);
        const xt::xarray<int> flatennedIndexs = xt::flatten(indexs);

        const std::vector<std::string> characters = readCharactersFile(charactersPath);
        const auto [text, processedIndexs] = greedyDecode(flatennedIndexs, characters);

        const xt::xarray<float> predsMaxProbs = xt::filter(xt::flatten(maxs), processedIndexs > 0);
        float confidenceScore = custom_mean(predsMaxProbs);

        return std::make_tuple(text, confidenceScore);
    }
    catch (std::exception &e)
    {
        std::cout << e.what() << std::endl;
    }

    return {};
}


int main()
{
    std::filesystem::path modelPath = "english_g2_dynamic_input.onnx";
    std::filesystem::path imgPath = "pre.png";
    std::filesystem::path charactersPath = "en.txt";

    // Create onnxrutime session
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_FATAL, "Easyocr");
    const Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    const Ort::AllocatorWithDefaultOptions allocator;

    Ort::Session *session = new Ort::Session(env, modelPath.generic_wstring().c_str(), sessionOptions);

    cv::Mat img = cv::imread(imgPath.generic_string());
    const auto [text, confidenceScore] = inference(session, img, charactersPath);

    std::cout << "Text: " << text << std::endl;
    std::cout << "Confidence: " << confidenceScore << std::endl;

    delete session;
    return 0;
}