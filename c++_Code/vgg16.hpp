#pragma once
#include <torch/torch.h>
struct VGG16Impl : torch::nn::Module {
    VGG16Impl(int num_classes);
    torch::Tensor forward(torch::Tensor x);
    
    torch::nn::Sequential features{nullptr};
    torch::nn::Sequential classifier{nullptr};
};
TORCH_MODULE(VGG16);

// vgg16.cpp
#include "vgg16.hpp"

VGG16Impl::VGG16Impl(int num_classes) {
    features = torch::nn::Sequential(
        // Conv Block 1
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        
        // Conv Block 2
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        
        // Conv Block 3
        torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        
        // Conv Block 4
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
        
        // Conv Block 5
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    );

    classifier = torch::nn::Sequential(
        torch::nn::Linear(512 * 7 * 7, 4096),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout(),
        torch::nn::Linear(4096, 4096),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout(),
        torch::nn::Linear(4096, num_classes)
    );

    register_module("features", features);
    register_module("classifier", classifier);
}

torch::Tensor VGG16Impl::forward(torch::Tensor x) {
    x = features->forward(x);
    x = x.view({x.size(0), -1});
    x = classifier->forward(x);
    return x;
}
