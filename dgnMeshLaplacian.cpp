#include "augmentation.h"
#include "diagnostics.h"
#include "hfe.h"
#include "meshlap.h"

#include <random>

using namespace aug;
using namespace hfe;

std::default_random_engine rnd;

void perturbMesh(const Geometry& in, const double stdDev, Geometry* out) {

    std::normal_distribution<double> normalDist(0.0, stdDev);

    // Make a copy of the mesh
    in.copyTo(out);

    // Perturb every vertex by gaussian noise
    for (auto v = out->getVertex(0); v.isValid(); v = v.nextById()) {
        v.ptrPosition()->x() += normalDist(rnd);
        v.ptrPosition()->y() += normalDist(rnd);
        v.ptrPosition()->z() += normalDist(rnd);
    }
}

struct MeshLaplacianParameters {
    Geometry mesh;
};
struct MeshLaplacianHyperparameters {
    double stdDev;
};
typedef MatrixParameterDistribution<MeshLaplacianParameters, MeshLaplacianHyperparameters> DistributionBase;
typedef ProblemDefinition<MeshLaplacianParameters, MeshLaplacianHyperparameters> ProblemDefType;

class MeshLaplacianDistribution : public DistributionBase {
public:
    void drawParameters(MeshLaplacianParameters* output) const override {
        perturbMesh(parameters.mesh, hyperparameters.stdDev, &output->mesh);
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const MeshLaplacianParameters& params) const override {
        Eigen::SparseMatrix<double> matrix;
        laplacian(params.mesh, &matrix);
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return parameters.mesh.vertexCount();
    }

    MeshLaplacianDistribution(MeshLaplacianParameters& parameters, MeshLaplacianHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<MeshLaplacianParameters, MeshLaplacianHyperparameters, MeshLaplacianDistribution>
        DiagnosticsType;
typedef ProblemRun<MeshLaplacianParameters, MeshLaplacianHyperparameters>
        ProblemRunType;

void dgnMeshLaplacian() {

    std::shared_ptr<Geometry> geo(loadJson("bunny.json"));


}