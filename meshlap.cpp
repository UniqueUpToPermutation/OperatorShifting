#include "meshlap.h"
#include "hfe.h"

#include <iostream>

using namespace Eigen;

namespace hfe {
    void massVector(const Geometry& geo, Eigen::VectorXd* output) {
        Eigen::VectorXd vec = Eigen::VectorXd::Zero(geo.vertexCount());

        // Each vertex has mass equal to a third of the area of its adjacent faces
        for (auto v = geo.constGetVertex(0); v.isValid(); v = v.nextById()) {
            double mass = 0.0;
            for (auto f = v.faces(); f.isValid(); f.next())
                mass += f().area();
            mass /= 3.0;
            vec(v.id()) = mass;
        }

        *output = vec;
    }

    void massMatrix(const Geometry &geo, Eigen::SparseMatrix<double> *output) {
        Eigen::VectorXd massVec;
        massVector(geo, &massVec);

        std::vector<Triplet<double>> nzEntries;
        nzEntries.reserve(geo.vertexCount());
        for (int i = 0, count = geo.vertexCount(); i < count; ++i)
            nzEntries.emplace_back(Triplet<double>(i, i, massVec(i)));

        *output = Eigen::SparseMatrix<double>(geo.vertexCount(), geo.vertexCount());
        output->setFromTriplets(nzEntries.begin(), nzEntries.end());
    }

    void enumLaplacian(const Geometry& geo, const Eigen::VectorXd& mass, std::vector<Eigen::Triplet<double>>* output) {
        output->reserve(geo.edgeCount() + geo.vertexCount());

        // For every vertex
        for (Vertex v = geo.constGetVertex(0); v.isValid(); v = v.nextById()) {

            double totalWeight = 0.0;
            // Get all outgoing edges of the vertex
            for (auto e_it = v.outgoing(); e_it.isValid(); e_it.next()) {
                Edge e = e_it();

                // Calculate cotangent of alpha
                double cotAlpha = 0.0;
                if (e.face().id() != -1) {
                    auto v1 = e.next().direction();
                    auto v2 = -e.next().next().direction();
                    cotAlpha = v1.dot(v2) / v1.cross(v2).norm();
                }

                // Calculate cotangent of beta
                double cotBeta = 0.0;
                if (e.opposite().face().id() != -1) {
                    auto v1 = e.opposite().next().direction();
                    auto v2 = -e.opposite().next().next().direction();
                    cotBeta = v1.dot(v2) / v1.cross(v2).norm();
                }

                double weight = (cotAlpha + cotBeta) / 2.0;

                totalWeight += weight;

                output->emplace_back(Triplet<double>(v.id(), e.head().id(), weight / mass(v.id())));
            }
            output->emplace_back(Triplet<double>(v.id(), v.id(), -totalWeight / mass(v.id())));
        }
    }

    void enumWeakLaplacian(const Geometry& geo, std::vector<Triplet<double>>* output) {
        VectorXd mass = VectorXd::Ones(geo.vertexCount());
        enumLaplacian(geo, mass, output);
    }

    void enumWeakLaplacianPositiveDefinite(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output) {
        VectorXd mass = -VectorXd::Ones(geo.vertexCount());
        enumLaplacian(geo, mass, output);
    }

    void weakLaplacian(const Geometry& geo, Eigen::SparseMatrix<double>* output) {
        std::vector<Triplet<double>> nzEntries;
        enumWeakLaplacian(geo, &nzEntries);

        output->resize(geo.vertexCount(), geo.vertexCount());
        output->setFromTriplets(nzEntries.begin(), nzEntries.end());
    }

    void weakLaplacianPositiveDefinite(const Geometry& geo, Eigen::SparseMatrix<double>* output) {
        std::vector<Triplet<double>> nzEntries;
        enumWeakLaplacianPositiveDefinite(geo, &nzEntries);

        output->resize(geo.vertexCount(), geo.vertexCount());
        output->setFromTriplets(nzEntries.begin(), nzEntries.end());
    }

    void enumLaplacian(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output) {
        Eigen::VectorXd mass;
        massVector(geo, &mass);
        enumLaplacian(geo, mass, output);
    }

    void enumLaplacianPositiveDefinite(const Geometry& geo, std::vector<Eigen::Triplet<double>>* output) {
        Eigen::VectorXd mass;
        massVector(geo, &mass);
        mass = -mass; // flip the signs of the mass
        enumLaplacian(geo, mass, output);
    }

    void laplacian(const Geometry& geo, SparseMatrix<double>* output) {
        std::vector<Eigen::Triplet<double>> nzEntries;
        enumLaplacian(geo, &nzEntries);
        output->resize(geo.vertexCount(), geo.vertexCount());
        output->setFromTriplets(nzEntries.begin(), nzEntries.end());
    }
}