#include <opshift/opshift.h>
#include <opshift/diagnostics.h>
#include <opshift/tests.h>

#include <random>
#include <cstring>

#define DEFAULT_N 64
#define DEFAULT_STD_DEV 0.5
#define DEFAULT_PERTURB_TYPE PERTURB_TYPE_DISCRETE
#define DEFAULT_PERTURB_TYPE_STRING "discrete"
#define DEFAULT_THREAD_COUNT 4
#define DEFAULT_NUM_SUB_RUNS_NAIVE 10000
#define DEFAULT_NUM_SUB_RUNS 100
#define SAMPLES_PER_SUB_RUN 100

using namespace opshift;

std::default_random_engine dgnGridLap2DRnd(std::chrono::system_clock::now().time_since_epoch().count());

void formLaplacian2D(const Eigen::MatrixXd& aHorizontal,
        const Eigen::MatrixXd& aVertical,
        Eigen::SparseMatrix<double>* output,
        const double h) {
    int n = aVertical.cols();
    int m = aHorizontal.rows();
    double h_sqrd = h * h;

    auto getVertId = [n, m](const int x, const int y) {
        if (x >= 0 && x < n && y >= 0 && y < m)
            return x + y * n;
        else
            return -1;
    };

    int nnz_upper_bound = n * m * 5;
    std::vector<Eigen::Triplet<double>> nonzeros;
    nonzeros.reserve(nnz_upper_bound);

    for (int x = 0; x < n; ++x) {
        for (int y = 0; y < m; ++y) {
            auto left_weight = aHorizontal(y, x) / h_sqrd;
            auto right_weight = aHorizontal(y, x + 1) / h_sqrd;
            auto up_weight = aVertical(y, x) / h_sqrd;
            auto down_weight = aVertical(y + 1, x) / h_sqrd;
            auto current_id = getVertId(x, y);
            auto left_id = getVertId(x - 1, y);
            auto right_id = getVertId(x + 1, y);
            auto up_id = getVertId(x, y - 1);
            auto down_id = getVertId(x, y + 1);

            Eigen::Triplet<double> left(current_id, left_id, -left_weight);
            Eigen::Triplet<double> right(current_id, right_id, -right_weight);
            Eigen::Triplet<double> up(current_id, up_id, -up_weight);
            Eigen::Triplet<double> down(current_id, down_id, -down_weight);
            auto degree = left_weight + right_weight + up_weight + down_weight;
            Eigen::Triplet<double> self(current_id, current_id, degree);

            if (left_id != -1)
                nonzeros.emplace_back(left);
            if (right_id != -1)
                nonzeros.emplace_back(right);
            if (up_id != -1)
                nonzeros.emplace_back(up);
            if (down_id != -1)
                nonzeros.emplace_back(down);
            nonzeros.emplace_back(self);
        }
    }

    *output = Eigen::SparseMatrix<double>(n * m, n * m);
    output->setFromTriplets(nonzeros.begin(), nonzeros.end());
}

void perturbBackground2D(const Eigen::MatrixXd& a, const PerturbType perturbType,
        const double std_dev, Eigen::MatrixXd* output) {
    size_t n = a.cols();
    size_t m = a.rows();

    switch (perturbType) {
        // Selects z_e from a discrete distribution { 1 - std_dev, 1 + std_dev }
        case PERTURB_TYPE_DISCRETE: {
            Eigen::MatrixXd rnd = Eigen::MatrixXd::Random(a.rows(), a.cols());
            for (size_t j = 0; j < m; ++j)
                for (size_t i = 0; i < n; ++i)
                    rnd(j, i) = (rnd(j, i) < 0.0 ? -1.0 : 1.0);

            Eigen::MatrixXd tmp = std_dev * rnd.array() + 1.0;
            *output = a.cwiseProduct(tmp);
            break;
        }
        // Selects z_e from a gamma distribution
        case PERTURB_TYPE_GAMMA: {
            auto alpha = 1.0 / (std_dev * std_dev);
            auto beta = std_dev * std_dev;
            std::gamma_distribution<double> gammaDist(alpha, beta);
            *output = Eigen::MatrixXd::Zero(m, n);
            for (size_t j = 0; j < m; ++j)
                for (size_t i = 0; i < n; ++i)
                (*output)(j, i) = a(j, i) * gammaDist(dgnGridLap2DRnd);
            break;
        }
    }

}

// For an n x n grid
struct GridLaplacian2DParameters {
    // Should be n x (n + 1), sampled between grid points
    Eigen::MatrixXd trueAHorizontal;
    // Should be (n + 1) x n, sampled between grid points
    Eigen::MatrixXd trueAVertical;
};
struct GridLaplacian2DHyperparameters {
    double stdDev;
    double h;
    PerturbType perturbType;
};
typedef MatrixParameterDistribution<GridLaplacian2DParameters, GridLaplacian2DHyperparameters> DistributionBase;
typedef ProblemDefinition<GridLaplacian2DParameters, GridLaplacian2DHyperparameters> ProblemDefType;

class GridLaplacian2DDistribution : public DistributionBase {
public:
    void drawParameters(GridLaplacian2DParameters* output) const override {
        perturbBackground2D(parameters.trueAHorizontal, hyperparameters.perturbType,
                hyperparameters.stdDev, &output->trueAHorizontal);
        perturbBackground2D(parameters.trueAVertical, hyperparameters.perturbType,
                hyperparameters.stdDev, &output->trueAVertical);
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GridLaplacian2DParameters& params) const override {
        Eigen::SparseMatrix<double> matrix;
        formLaplacian2D(params.trueAHorizontal, params.trueAVertical, &matrix, hyperparameters.h);
        return std::shared_ptr<IInvertibleMatrixOperator>(new SparseMatrixSampleSPD(matrix));
    }
    size_t getDimension() const override {
        return parameters.trueAHorizontal.rows() * parameters.trueAVertical.cols();
    }
    bool isSPD() const override {
        return true;
    }

    GridLaplacian2DDistribution(GridLaplacian2DParameters& parameters, GridLaplacian2DHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GridLaplacian2DParameters, GridLaplacian2DHyperparameters, GridLaplacian2DDistribution>
        DiagnosticsType;
typedef ProblemRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>
        ProblemRunType;

void dgnGridLaplacian2D(int argc, char** argv) {
    size_t n = DEFAULT_N;
    double std_dev = DEFAULT_STD_DEV;
    std::string perturbTypeString = DEFAULT_PERTURB_TYPE_STRING;
    int threadCount = DEFAULT_THREAD_COUNT;
    int numSubRunsNaive = DEFAULT_NUM_SUB_RUNS_NAIVE;
    int numSubRuns = DEFAULT_NUM_SUB_RUNS;
    int samplesPerSubRun = SAMPLES_PER_SUB_RUN;

    std::cout << "################ GridLaplacian2D ################" << std::endl << std::endl;

    // Read global configuration from file
    if (globalConfigAvailable) {
        auto config = globalConfig["GridLaplacian1D"];
        n = config["gridSize"].get<int>();
        std_dev = config["stdDev"].get<double>();
        perturbTypeString = config["distribution"].get<std::string>();
        threadCount = config["threadCount"].get<int>();
        numSubRunsNaive = config["numSubRunsNaive"].get<int>();
        numSubRuns = config["numSubRuns"].get<int>();
        samplesPerSubRun = config["samplesPerSubRun"].get<int>();

        std::cout << "Configuration:" << std::endl;
        std::cout << std::setw(4) << config << std::endl << std::endl;
    }

    if (argc > 0)
        n = std::stoi(argv[0]);
    if (argc > 1)
        std_dev = std::stod(argv[1]);
    if (argc > 2)
        perturbTypeString = argv[2];

    std::for_each(perturbTypeString.begin(), perturbTypeString.end(), [](char& c) { c = std::tolower(c); });

    PerturbType perturbType = DEFAULT_PERTURB_TYPE;

    if (perturbTypeString == "discrete")
        perturbType = PERTURB_TYPE_DISCRETE;
    else if (perturbTypeString == "gamma")
        perturbType = PERTURB_TYPE_GAMMA;
    else
        perturbType = DEFAULT_PERTURB_TYPE;

    double h = 1.0 / (n + 1.0);

    Eigen::MatrixXd true_a_horiz = Eigen::MatrixXd::Ones(n, n + 1);
    Eigen::MatrixXd true_a_vert = Eigen::MatrixXd::Ones(n + 1, n);
    Eigen::VectorXd eqSpaced = Eigen::VectorXd::LinSpaced(n, 1, n) * h;
    Eigen::VectorXd oneVec = Eigen::VectorXd::Ones(n);
    Eigen::MatrixXd xs_mat = (oneVec * eqSpaced.transpose()).transpose();
    Eigen::MatrixXd ys_mat = (eqSpaced * oneVec.transpose()).transpose();
    Eigen::VectorXd xs = Eigen::Map<Eigen::VectorXd>(xs_mat.data(), xs_mat.rows() * xs_mat.cols());
    Eigen::VectorXd ys = Eigen::Map<Eigen::VectorXd>(ys_mat.data(), ys_mat.rows() * ys_mat.cols());

    std::function<void(Eigen::VectorXd *)> bDistribution_func = [&xs, &ys](Eigen::VectorXd *output) {
        *output = Eigen::cos(2.0 * M_PI * xs.array()).cwiseProduct(Eigen::cos(2.0 * M_PI * ys.array()));
    };
    auto bDistribution = std::shared_ptr<IVectorDistribution>(new VectorDistributionFromLambda(bDistribution_func));

    auto params = GridLaplacian2DParameters{true_a_horiz, true_a_vert};
    auto hyperparams = GridLaplacian2DHyperparameters{std_dev, h, perturbType};
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GridLaplacian2DDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    // problem_def->bDistribution = bDistribution; // Don't use the above distribution, use random normal by default
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRunsNaive;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new OpshiftRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftRun<GridLaplacian2DParameters, GridLaplacian2DHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRebasedAccelRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRebasedAccelRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyOpshiftTruncatedRebasedAccelRun<GridLaplacian2DParameters,
                    GridLaplacian2DHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    diagnostics.run(threadCount);
    diagnostics.printResults();
    diagnostics.printLatexTable();
}

int main(int argc, char *argv[]) {
    loadConfig();
    dgnGridLaplacian2D(argc - 1, &argv[1]);
}