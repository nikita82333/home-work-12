#pragma once

#include <vector>
#include <cstddef>

#include <Eigen/Dense>

namespace mnist {

class Classifier {
public:
    using features_t = std::vector<float>;
    using probas_t = std::vector<float>;

    virtual ~Classifier() {}

    [[nodiscard]] virtual size_t num_classes() const = 0;

    [[nodiscard]] virtual size_t predict(const features_t&) const = 0;

    [[nodiscard]] virtual probas_t predict_proba(const features_t&) const = 0;
};

class MlpClassifier: public Classifier {
public:
    MlpClassifier(Eigen::MatrixXf, Eigen::MatrixXf);

    [[nodiscard]] size_t num_classes() const override;

    [[nodiscard]] size_t predict(const features_t&) const override;

    [[nodiscard]] probas_t predict_proba(const features_t&) const override;

private:
    Eigen::MatrixXf w1_, w2_;
};

}