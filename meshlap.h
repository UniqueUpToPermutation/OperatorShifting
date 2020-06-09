//
// Created by Phil on 6/7/20.
//

#ifndef OPERATORAUGMENTATION_MESHLAP_H
#define OPERATORAUGMENTATION_MESHLAP_H

#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace hfe {
    class Geometry;

    void enumWeakLaplacian(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output);
    void enumWeakLaplacianPositiveDefinite(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output);
    void enumLaplacian(const Geometry& geo, const Eigen::VectorXd& mass, std::vector<Eigen::Triplet<double>>* output);
    void enumLaplacian(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output);
    void enumLaplacianPositiveDefinite(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output);
    void weakLaplacian(const Geometry& geo, Eigen::SparseMatrix<double>* output);
    void weakLaplacianPositiveDefinite(const Geometry& geo, Eigen::SparseMatrix<double>* output);
    void massVector(const Geometry& geo, Eigen::VectorXd* output);
    void massMatrix(const Geometry& geo, Eigen::SparseMatrix<double>* output);
    void laplacian(const Geometry& geo, Eigen::SparseMatrix<double>* output);
    void laplacianPositiveDefinite(const Geometry& geo, Eigen::SparseMatrix<double>* output);
}

#endif //OPERATORAUGMENTATION_MESHLAP_H
