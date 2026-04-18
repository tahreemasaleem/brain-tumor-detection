#pragma once
#include <torch/torch.h>
#include "vgg16.hpp"
#include "dataset.hpp"

class Trainer {
public:
    Trainer(const std::string& train_dir, const std::string& val_dir);
    void train(int num_epochs);
    void save_model(const std::string& path);

private:
    torch::Device device_;
    VGG16 model_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<torch::data::DataLoader<TumorDataset>> train_loader_;
    std::unique_ptr<torch::data::DataLoader<TumorDataset>> val_loader_;
};

// trainer.cpp
#include "trainer.hpp"
#include <iostream>

Trainer::Trainer(const std::string& train_dir, const std::string& val_dir) 
    : device_(torch::kCUDA), model_(4) {
    
    model_->to(device_);
    optimizer_ = std::make_unique<torch::optim::Adam>(
        model_->parameters(), torch::optim::AdamOptions(1e-4));

    auto train_dataset = TumorDataset(train_dir, device_);
    auto val_dataset = TumorDataset(val_dir, device_);

    train_loader_ = std::make_unique<torch::data::DataLoader<TumorDataset>>(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(32).workers(4)
    );

    val_loader_ = std::make_unique<torch::data::DataLoader<TumorDataset>>(
        std::move(val_dataset),
        torch::data::DataLoaderOptions().batch_size(32).workers(4)
    );
}

void Trainer::train(int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Training phase
        model_->train();
        double train_loss = 0.0;
        int correct = 0;
        int total = 0;

        for (auto& batch : *train_loader_) {
            auto data = batch.data;
            auto target = batch.target;

            optimizer_->zero_grad();
            auto output = model_->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            
            loss.backward();
            optimizer_->step();

            train_loss += loss.item<float>();
            auto pred = output.argmax(1);
            correct += pred.eq(target).sum().item<int64_t>();
            total += target.size(0);
        }

        // Validation phase
        model_->eval();
        double val_loss = 0.0;
        int val_correct = 0;
        int val_total = 0;

        torch::NoGradGuard no_grad;
        for (auto& batch : *val_loader_) {
            auto data = batch.data;
            auto target = batch.target;

            auto output = model_->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);

            val_loss += loss.item<float>();
            auto pred = output.argmax(1);
            val_correct += pred.eq(target).sum().item<int64_t>();
            val_total += target.size(0);
        }

        std::cout << "Epoch: " << epoch + 1 << "/" << num_epochs << "\n";
        std::cout << "Training Loss: " << train_loss / total 
                  << " Accuracy: " << 100.0 * correct / total << "%\n";
        std::cout << "Validation Loss: " << val_loss / val_total 
                  << " Accuracy: " << 100.0 * val_correct / val_total << "%\n\n";
    }
}

void Trainer::save_model(const std::string& path) {
    torch::save(model_, path);
}
