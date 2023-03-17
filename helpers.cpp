#include "helpers.h"

#include <fstream>
#include <sstream>
#include <iterator>

namespace mnist {

    Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& stream) {
        Eigen::MatrixXf res(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float val;
                stream >> val;
                res(i, j) = val;
            }
        }
        return res;
    }

    Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string& filepath) {
        std::ifstream stream{filepath};
        return read_mat_from_stream(rows, cols, stream);
    }

    bool read_features(std::istream& stream, Classifier::features_t& features) {
        std::string line;
        std::getline(stream, line);

        features.clear();
        std::istringstream linestream{line};
        double value;
        while (linestream >> value) {
            features.push_back(value);
        }
        return stream.good();
    }

    Classifier::features_t split_to_float(const std::string &str, char d)
    {
        Classifier::features_t result;

        std::string::size_type start = 0;
        std::string::size_type stop = str.find_first_of(d);
        while (stop != std::string::npos)
        {
            result.push_back(std::stof(str.substr(start, stop - start)));

            start = stop + 1;
            stop = str.find_first_of(d, start);
        }

        result.push_back(std::stof(str.substr(start)));

        return result;
    }

    bool read_features_csv(std::istream& stream, Classifier::features_t& features) {
        std::string line;
        std::getline(stream, line);

        features.clear();
        if (!line.empty()) {
            features = split_to_float(line, ',');
        }
        return stream.good();
    }

    std::vector<float> read_vector(std::istream& stream) {
        std::vector<float> result;

        std::copy(std::istream_iterator<float>(stream),
                  std::istream_iterator<float>(),
                  std::back_inserter(result));
        return result;
    }

}