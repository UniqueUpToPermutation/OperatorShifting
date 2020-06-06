#include "augmentation.h"
#include "diagnostics.h"

using namespace aug;

void formLaplacian2D(const Eigen::VectorXd& a, Eigen::SparseMatrix<double>* output) {
    int n = a.size();
    double h = 1.0/((double)n);
    double h_sqrd = h * h;
}

void perturbBackground1D(const Eigen::VectorXd& a, const double std_dev, Eigen::VectorXd* output) {
    size_t n = a.size();
    Eigen::VectorXd rnd = Eigen::VectorXd::Random(a.size());
    for (size_t i = 0; i < n; ++i)
        rnd(i) = (rnd(i) < 0.0 ? -1.0 : 1.0);
    auto tmp = std_dev * rnd.array() + 1.0;
    *output = a.cwiseProduct(tmp.matrix());
}

struct GridLaplacian1DParameters {
    Eigen::VectorXd trueA;
};
struct GridLaplacian1DHyperparameters {
    double stdDev;
};
typedef MatrixParameterDistribution<GridLaplacian1DParameters, GridLaplacian1DHyperparameters> DistributionBase;
typedef ProblemDefinition<GridLaplacian1DParameters, GridLaplacian1DHyperparameters> ProblemDefType;

class GridLaplacian1DDistribution : public DistributionBase {
public:
    void drawParameters(GridLaplacian1DParameters* output) const override {
        perturbBackground1D(parameters.trueA, hyperparameters.stdDev, &output->trueA);
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GridLaplacian1DParameters& params) const override {
        Eigen::SparseMatrix<double> matrix;
        formLaplacian2D(params.trueA, &matrix);
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return parameters.trueA.size() - 1u;
    }

    GridLaplacian1DDistribution(GridLaplacian1DParameters& parameters, GridLaplacian1DHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GridLaplacian1DParameters, GridLaplacian1DHyperparameters, GridLaplacian1DDistribution>
        DiagnosticsType;
typedef ProblemRun<GridLaplacian1DParameters, GridLaplacian1DHyperparameters>
        ProblemRunType;