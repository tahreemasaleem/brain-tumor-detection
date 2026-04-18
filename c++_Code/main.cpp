#include "trainer.hpp"
#include <iostream>

int main() {
    try {
        Trainer trainer("path/to/train/data", "path/to/val/data");
        trainer.train(50);  // 50 epochs
        trainer.save_model("tumor_classifier.pt");
        std::cout << "Model saved successfully!\n";
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
