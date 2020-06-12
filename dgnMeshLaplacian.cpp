#include "auxDiagnostics.h"
#include "hfe.h"
#include "meshlap.h"

#include <random>

using namespace aug;
using namespace hfe;

std::default_random_engine rnd(std::chrono::system_clock::now().time_since_epoch().count());

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
    double gamma;
};
typedef MatrixParameterDistribution<MeshLaplacianParameters, MeshLaplacianHyperparameters> DistributionBase;
typedef ProblemDefinition<MeshLaplacianParameters, MeshLaplacianHyperparameters> ProblemDefType;

class MeshLaplacianDistribution : public DistributionBase {
public:
    void drawParameters(MeshLaplacianParameters* output) const override {
        perturbMesh(parameters.mesh, hyperparameters.stdDev, &output->mesh);
    }

    std::shared_ptr<IInvertibleMatrixOperator> convert(const MeshLaplacianParameters& params) const override {
        Eigen::SparseMatrix<double> weakLap;
        weakLaplacianPositiveDefinite(params.mesh, &weakLap);
        Eigen::SparseMatrix<double> massMat;
        massMatrix(params.mesh, &massMat);
        Eigen::SparseMatrix<double> result = hyperparameters.gamma * massMat + weakLap;
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(result));
    }

    std::shared_ptr<IMatrixOperator> convertAuxiliary(const MeshLaplacianParameters &params) const override {
        Eigen::SparseMatrix<double> massMat;
        massMatrix(params.mesh, &massMat);
        return std::shared_ptr<IMatrixOperator>(new SparseMatrixSampleNonInvertible(massMat));
    }

    size_t getDimension() const override {
        return parameters.mesh.vertexCount();
    }

    MeshLaplacianDistribution(MeshLaplacianParameters& parameters, MeshLaplacianHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters, true) {}
};

typedef Diagnostics<MeshLaplacianParameters, MeshLaplacianHyperparameters, MeshLaplacianDistribution>
        DiagnosticsType;
typedef ProblemRun<MeshLaplacianParameters, MeshLaplacianHyperparameters>
        ProblemRunType;

void dgnMeshLaplacian() {
    std::shared_ptr<Geometry> geo(loadJson("Meshes/bunny_low.json"));

    double scaling = (geo->getBoundingBox().upper - geo->getBoundingBox().lower).norm();

    double stddev = 0.0025;
    double gamma = 0.1; // Make system invertible by shifting
    stddev *= scaling;

    MeshLaplacianParameters trueParams;
    geo->copyTo(&trueParams.mesh); // Copy mesh to parameters
    MeshLaplacianHyperparameters hyperParams{stddev, gamma};
    auto trueDistribution = std::shared_ptr<DistributionBase>(
            new MeshLaplacianDistribution(trueParams, hyperParams));
    auto problemDef = std::shared_ptr<ProblemDefType>(
            new ProblemDefType(trueDistribution));
    DiagnosticsType diagnostics(problemDef);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<MeshLaplacianParameters, MeshLaplacianHyperparameters>(problemDef.get()));
    run->numberSubRuns = 10000;
    diagnostics.addRun(run);

    // Op Augmentation
    run = std::shared_ptr<ProblemRunType>(
            new AuxAugmentationRun<MeshLaplacianParameters, MeshLaplacianHyperparameters>(problemDef.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    // Op En Augmentation
    run = std::shared_ptr<ProblemRunType>(
            new AuxEnergyAugmentationRun<MeshLaplacianParameters, MeshLaplacianHyperparameters>(problemDef.get()));
    run->numberSubRuns = 100;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    diagnostics.run(4);
    diagnostics.printResults();
    diagnostics.printLatexTable();
}