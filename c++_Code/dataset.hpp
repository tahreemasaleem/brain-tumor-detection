#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <vector>

class TumorDataset : public torch::data::Dataset<TumorDataset> {
private:
    std::vector<std::string> image_paths_;
    std::vector<int> labels_;
    torch::Device device_;

public:
    TumorDataset(const std::string& data_dir, torch::Device device);
    torch::data::Example<> get(size_t index) override;
    torch::optional<size_t> size() const override;
};

// dataset.cpp
#include "dataset.hpp"

TumorDataset::TumorDataset(const std::string& data_dir, torch::Device device) : device_(device) {
    std::vector<std::string> class_names = {"class1", "class2", "class3", "class4"};
    
    for (size_t class_idx = 0; class_idx < class_names.size(); ++class_idx) {
        std::string class_path = data_dir + "/" + class_names[class_idx];
        for (const auto& entry : std::filesystem::directory_iterator(class_path)) {
            image_paths_.push_back(entry.path().string());
            labels_.push_back(class_idx);
        }
    }
}

torch::data::Example<> TumorDataset::get(size_t index) {
    std::string image_path = image_paths_[index];
    int label = labels_[index];

    cv::Mat image = cv::imread(image_path);
    cv::resize(image, image, cv::Size(224, 224));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    torch::Tensor img_tensor = torch::from_blob(image.data, {224, 224, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
    img_tensor = img_tensor.to(device_);
    
    img_tensor = torch::normalize(img_tensor, 
        {0.485, 0.456, 0.406}, 
        {0.229, 0.224, 0.225});

    return {img_tensor, torch::tensor(label, device_)};
}

torch::optional<size_t> TumorDataset::size() const {
    return image_paths_.size();
}