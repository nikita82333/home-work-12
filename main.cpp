#include <iostream>
#include <fstream>

#include "mlp_classifier.h"
#include "helpers.h"

using namespace mnist;

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: fashion_mnist <test_file>" << std::endl;
        std::cerr << " - <test_file> - test sample in csv format" << std::endl;
        return 1;
    }

    std::ifstream test_data{argv[1]};
    if (!test_data.is_open()) {
        std::cerr << "Source file \"" << argv[1] << "\" is not exists." << std::endl;
        return 1;
    }

    const size_t input_dim = 784;
    const size_t hidden_dim = 128;
    const size_t output_dim = 10;

    auto w1 = read_mat_from_file(input_dim, hidden_dim, "train/w1.txt");
    auto w2 = read_mat_from_file(hidden_dim, output_dim, "train/w2.txt");

    auto clf = MlpClassifier{w1.transpose(), w2.transpose()};
    auto features = MlpClassifier::features_t{};

    size_t true_count = 0, total_count = 0;
    for (;;) {
        try {
            if (!read_features_csv(test_data, features)) {
                break;
            }
            ++total_count;
            size_t y_true;
            y_true = static_cast<size_t>(features[0]);
            features.erase(features.begin());

            if (y_true == clf.predict(features)) {
                ++true_count;
            }
        } catch (std::exception& exception) {
            std::cerr << "Incorrect data in " << total_count + 1 << "-th line of the source file. ";
            std::cerr << "The line was ignored." << std::endl;
        }
    }

    std::cout << "Model accuracy = " << static_cast<double>(true_count) / total_count << std::endl;

    return 0;
}
