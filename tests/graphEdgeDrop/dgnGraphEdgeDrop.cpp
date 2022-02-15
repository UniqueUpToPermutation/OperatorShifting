#include <opshift/diagnostics.h>
#include <opshift/opshift.h>
#include <opshift/graphlap.h>
#include <opshift/matutil.h>
#include <opshift/tests.h>

#include <random>
#include <set>
#include <cstring>

using namespace opshift;
using namespace lemon;

#define DEFAULT_P 0.75
#define DEFAULT_GAMMA 1.0
#define DEFAULT_GRAPH_PATH "Graphs/fb-pages-food/fb-pages-food.edges"
#define DEFAULT_PERTURB_TYPE_STRING "gamma"
#define DEFAULT_FORMAT "unweighted"
#define DEFAULT_PERTURB_TYPE PERTURB_TYPE_GAMMA
#define DEFAULT_THREAD_COUNT 4
#define DEFAULT_NUM_SUB_RUNS_NAIVE 10000
#define DEFAULT_NUM_SUB_RUNS 100
#define SAMPLES_PER_SUB_RUN 100

std::default_random_engine dgnEdgeDropRnd(std::chrono::system_clock::now().time_since_epoch().count());

void subsample(const ListGraph* graph, const ListGraph::EdgeMap<double>* base,
               const double p,
               ListGraph::EdgeMap<double>* output) {
    std::bernoulli_distribution bernDist(p);

    for (ListGraph::EdgeIt e(*graph); e != INVALID; ++e) {
        double oldValue = base->operator[](e);
        double newValue = bernDist(dgnEdgeDropRnd) * oldValue / p;
        output->set(e, newValue);
    }
}

struct GraphEdgeDropParameters {
    std::shared_ptr<ListGraph::EdgeMap<double>> weights;
};
struct GraphEdgeDropHyperparameters {
    ListGraph* graph;
    double p;
    double gamma;
};
typedef MatrixParameterDistribution<GraphEdgeDropParameters, GraphEdgeDropHyperparameters > DistributionBase;
typedef ProblemDefinition<GraphEdgeDropParameters, GraphEdgeDropHyperparameters > ProblemDefType;

class GraphEdgeDropDistribution : public DistributionBase {
public:
    void drawParameters(GraphEdgeDropParameters* output) const override {
        auto newWeights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(*hyperparameters.graph));
        subsample(hyperparameters.graph, parameters.weights.get(), hyperparameters.p, newWeights.get());
        output->weights = newWeights;
    }
    std::shared_ptr<IInvertibleMatrixOperator> convert(const GraphEdgeDropParameters& params) const override {
        Eigen::SparseMatrix<double> matrix(getDimension(), getDimension());
        graphLaplacian(hyperparameters.graph, params.weights.get(), &matrix);
        int dim = getDimension();
        // Add stuff to the diagonal to better conditioned system
        for (int i = 0; i < dim; ++i)
            matrix.coeffRef(i, i) = matrix.coeff(i, i) + hyperparameters.gamma;
        return std::shared_ptr<IInvertibleMatrixOperator>(new SparseMatrixSampleSPD(matrix));
    }
    size_t getDimension() const override {
        return countNodes(*hyperparameters.graph);
    }
    bool isSPD() const override {
        return true;
    }
    GraphEdgeDropDistribution(GraphEdgeDropParameters& parameters, GraphEdgeDropHyperparameters& hyperparameters) :
            DistributionBase(parameters, hyperparameters) {}
};

typedef Diagnostics<GraphEdgeDropParameters, GraphEdgeDropHyperparameters, GraphEdgeDropDistribution>
        DiagnosticsType;
typedef ProblemRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>
        ProblemRunType;

void dgnGraphEdgeDrop(int argc, char** argv) {
    double p = DEFAULT_P;
    double gamma = DEFAULT_GAMMA;

    std::string format = DEFAULT_FORMAT;
    std::string graphPath =  DEFAULT_GRAPH_PATH;
    int threadCount = DEFAULT_THREAD_COUNT;
    int numSubRunsNaive = DEFAULT_NUM_SUB_RUNS_NAIVE;
    int numSubRuns = DEFAULT_NUM_SUB_RUNS;
    int samplesPerSubRun = SAMPLES_PER_SUB_RUN;

    std::cout << "################ GraphEdgeDrop ################" << std::endl << std::endl;

    // Read global configuration from file
    if (globalConfigAvailable) {
        auto config = globalConfig["GraphEdgeDrop"];
        format = config["format"].get<std::string>();
        graphPath = config["graphSrc"].get<std::string>();
        p = config["p"].get<double>();
        gamma = config["gamma"].get<double>();
        threadCount = config["threadCount"].get<int>();
        numSubRunsNaive = config["numSubRunsNaive"].get<int>();
        numSubRuns = config["numSubRuns"].get<int>();
        samplesPerSubRun = config["samplesPerSubRun"].get<int>();

        std::cout << "Configuration:" << std::endl;
        std::cout << std::setw(4) << config << std::endl << std::endl;
    }

    if (argc > 0)
        graphPath = argv[0];
    if (argc > 1)
        format = argv[1];
    if (argc > 2)
        p = std::stod(argv[2]);
    if (argc > 3)
        gamma = std::stod(argv[3]);

    std::for_each(format.begin(), format.end(), [](char& c) { c = std::tolower(c); });

    ListGraph graph;
    GraphEdgeDropParameters params;
    params.weights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(graph, 1.0));

    if (format == "weighted") {
        loadGraphWeighted(graphPath, &graph, params.weights.get());
    } else {
        loadGraphUnweighted(graphPath, &graph);
        params.weights = std::shared_ptr<ListGraph::EdgeMap<double>>(new ListGraph::EdgeMap<double>(graph, 1.0));
    }

    auto hyperparams = GraphEdgeDropHyperparameters{&graph, p, gamma};
    auto true_mat_dist = std::shared_ptr<DistributionBase>(new GraphEdgeDropDistribution(params, hyperparams));
    auto problem_def = std::shared_ptr<ProblemDefType>(new ProblemDefType(true_mat_dist));
    auto diagnostics = DiagnosticsType(problem_def);

    // Naive run
    auto run = std::shared_ptr<ProblemRunType>(
            new NaiveRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRunsNaive;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AugmentationRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new EnergyAugmentationRun<GraphEdgeDropParameters, GraphEdgeDropHyperparameters>(problem_def.get()));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 2, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new TruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 4, TRUNCATION_WINDOW_HARD));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 2));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 4));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = samplesPerSubRun;
    diagnostics.addRun(run);

    run = std::shared_ptr<ProblemRunType>(
            new AccelShiftTruncatedEnergyAugmentationRun<GraphEdgeDropParameters,
                    GraphEdgeDropHyperparameters>(problem_def.get(), 6));
    run->numberSubRuns = numSubRuns;
    run->samplesPerSubRun = 100;
    diagnostics.addRun(run);

    diagnostics.run(threadCount);
    diagnostics.printResults();
    diagnostics.printLatexTable();
}

int main(int argc, char *argv[]) {
    loadConfig();
    dgnGraphEdgeDrop(argc - 1, &argv[1]);
}