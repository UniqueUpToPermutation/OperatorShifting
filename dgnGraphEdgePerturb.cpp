#include "diagnostics.h"
#include "augmentation.h"
#include "graphlap.h"
#include "matutil.h"
#include "testbeds.h"

#include <random>
#include <set>
#include <cstring>

#define DEFAULT_STD_DEV 2.0
#define DEFAULT_GRAPH_PATH "Graphs/fb-pages-food/fb-pages-food.edges"
#define DEFAULT_PERTURB_TYPE_STRING "gamma"
#define DEFAULT_FORMAT "unweighted"
#define DEFAULT_PERTURB_TYPE PERTURB_TYPE_GAMMA
#define DEFAULT_THREAD_COUNT 4
#define DEFAULT_NUM_SUB_RUNS_NAIVE 10000
#define DEFAULT_NUM_SUB_RUNS 100
#define SAMPLES_PER_SUB_RUN 100

using namespace aug;
using namespace lemon;

std::default_random_engine dgnEdgePeturbRnd(std::chrono::system_clock::now().time_since_epoch().count());

void perturb(const ListGraph* graph, const ListGraph::EdgeMap<double>* base,
             const PerturbType pertubType, const double stddev,
             ListGraph::EdgeMap<double>* output) {
    switch (pertubType) {
        case PERTURB_TYPE_DISCRETE: {
            std::bernoulli_distribution bernoulliDist;

            for (ListGraph::EdgeIt e(*graph); e != INVALID; ++e) {
                double oldValue = base->operator[](e);
                double newValue = ((stddev * (2.0 * bernoulliDist(dgnEdgePeturbRnd) - 1.0)) + 1.0) * oldValue;
                output->set(e, newValue);
            }
            break;
        }
        case PERTURB_TYPE_GAMMA: {
            auto alpha = 1.0 / (stddev * stddev);
            auto beta = stddev * stddev;
            std::gamma_distribution<double> gammaDist(alpha, beta);

            for (ListGraph::EdgeIt e(*graph); e != INVALID; ++e) {
                double oldValue = base->operator[](e);
                double newValue = gammaDist(dgnEdgePeturbRnd) * oldValue;
                output->set(e, newValue);
            }
            break;
        }
    }
}

struct GraphEdgePerturbParameters {
    std::shared_ptr<ListGraph::EdgeMap<double>> weights;
};
struct GraphEdgePerturbHyperparameters {
    ListGraph* graph;
    Eigen::SparseMatrix<double>* interiorExtractorLeft;
    Eigen::SparseMatrix<double>* interiorExtractorRight;
    double stddev;
    PerturbType perturbType;
};
typedef MatrixParameterDistribution<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters > DistributionBase;
typedef ProblemDefinition<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters > ProblemDefType;

class GraphEdgePerturbDistribution : public DistributionBase {
public:
    void drawParameters(GraphEdgePerturbParameters* output) const override {
        auto newWeights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(*hyperparameters.graph));
        perturb(hyperparameters.graph, parameters.weights.get(), hyperparameters.perturbType,
                hyperparameters.stddev, newWeights.get());
        output->weights = newWeights;
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GraphEdgePerturbParameters& params) const override {
        Eigen::SparseMatrix<double> matrix(getDimension(), getDimension());
        graphLaplacian(hyperparameters.graph, params.weights.get(), &matrix);
        matrix = (*hyperparameters.interiorExtractorLeft) * matrix * (*hyperparameters.interiorExtractorRight);
        return std::shared_ptr<IInvertibleMatrixOperator>(new DefaultSparseMatrixSample(matrix));
    }
    size_t getDimension() const override {
        return hyperparameters.interiorExtractorLeft->rows();
    }
    GraphEdgePerturbDistribution(GraphEdgePerturbParameters& parameters, GraphEdgePerturbHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters, GraphEdgePerturbDistribution>
        DiagnosticsType;
typedef ProblemRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>
        ProblemRunType;

void dgnGraphEdgePerturb(int argc, char** argv) {
    double std_dev = DEFAULT_STD_DEV;
    std::string format = DEFAULT_FORMAT;
    std::string graphPath =  DEFAULT_GRAPH_PATH;
    std::string perturbTypeString = DEFAULT_PERTURB_TYPE_STRING;
    int threadCount = DEFAULT_THREAD_COUNT;
    int numSubRunsNaive = DEFAULT_NUM_SUB_RUNS_NAIVE;
    int numSubRuns = DEFAULT_NUM_SUB_RUNS;
    int samplesPerSubRun = SAMPLES_PER_SUB_RUN;
    std::vector<int> boundary = {1, 60, 100, 127, 200}; // Arbitrarily selected nodes

    std::cout << "################ GraphEdgePerturb ################" << std::endl << std::endl;

    // Read global configuration from file
    if (globalConfigAvailable) {
        auto config = globalConfig["GraphEdgePerturb"];
        format = config["format"].get<std::string>();
        graphPath = config["graphSrc"].get<std::string>();
        std_dev = config["stdDev"].get<double>();
        perturbTypeString = config["distribution"].get<std::string>();
        threadCount = config["threadCount"].get<int>();
        numSubRunsNaive = config["numSubRunsNaive"].get<int>();
        numSubRuns = config["numSubRuns"].get<int>();
        samplesPerSubRun = config["samplesPerSubRun"].get<int>();
        boundary = config["boundary"].get<std::vector<int>>();

        std::cout << "Configuration:" << std::endl;
        std::cout << std::setw(4) << config << std::endl << std::endl;
    }

    if (argc > 0)
        graphPath = argv[0];
    if (argc > 1)
        format = argv[1];
    if (argc > 2)
        std_dev = std::stod(argv[2]);
    if (argc > 3)
        perturbTypeString = argv[3];

    std::for_each(format.begin(), format.end(), [](char& c) { c = std::tolower(c); });
    std::for_each(perturbTypeString.begin(), perturbTypeString.end(), [](char& c) { c = std::tolower(c); });

    PerturbType perturbType = DEFAULT_PERTURB_TYPE;

    if (perturbTypeString == "discrete")
        perturbType = PERTURB_TYPE_DISCRETE;
    else if (perturbTypeString == "gamma")
        perturbType = PERTURB_TYPE_GAMMA;
    else
        perturbType = DEFAULT_PERTURB_TYPE;

    ListGraph graph;
    GraphEdgePerturbParameters params;
    params.weights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(graph, 1.0));

    if (format == "weighted") {
        loadGraphWeighted(graphPath, &graph, params.weights.get());
    }
    else {
        loadGraphUnweighted(graphPath, &graph);
        params.weights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(graph, 1.0));
    }


    std::vector<int> all;
    int nodeCount = countNodes(graph);
    all.reserve(nodeCount);
    for (int i = 0; i < nodeCount; ++i)
        all.emplace_back(i);
    std::vector<int> interior;
    std::set_difference(all.begin(), all.end(), boundary.begin(), boundary.end(),
                        std::inserter(interior, interior.begin()));
    Eigen::SparseMatrix<double> interiorExtractorLeft;
    createSlicingMatrix(all.size(), interior, &interiorExtractorLeft);
    Eigen::SparseMatrix<double> interiorExtractorRight = interiorExtractorLeft.transpose();

    auto hyperparams = GraphEdgePerturbHyperparameters{&graph, &interiorExtractorLeft, &interiorExtractorRight,
                                                       std_dev, perturbType};
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GraphEdgePerturbDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRunsNaive;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AugmentationRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyAugmentationRun<GraphEdgePerturbParameters, GraphEdgePerturbHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgePerturbParameters,
                    GraphEdgePerturbHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    diagnostics.run(threadCount);
    diagnostics.printResults();
    diagnostics.printLatexTable();
}