#include "matutil.h"

void createSlicingMatrix(const int cols, const std::vector<int>& indices, Eigen::SparseMatrix<double>* out) {
    out->resize(indices.size(), cols);
    std::vector<Eigen::Triplet<double>> nz;
    nz.reserve(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
        nz.emplace_back(Eigen::Triplet<double>(i, indices[i], 1.0));
    out->setFromTriplets(nz.begin(), nz.end());
}