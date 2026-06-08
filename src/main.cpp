#include <iostream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <sstream>

#include <vector>
#include <cctype>
#include <cstdint>
#include <cmath>      
#include <algorithm>  

namespace ksub {
    bool is_non_monotone = false;
}

#include <sys/resource.h>
#include <sys/stat.h>

#include "mygraph.h"
#include "kfunctions.h"
#include "kfunctions_impl.h"

#include "matroid.h" 

#include "algs/result.h"
#include "algs/simple_greedy.h"
#include "algs/fair_greedy.h"
#include "algs/k_greedy_is.h"
#include "algs/k_greedy_ts.h"  
#include "algs/fksm_alg.h"
#include "algs/streaming_1consistent_ksubmodular_matroid.h"

static long getPeakRSS_KB() {
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r) == 0) return r.ru_maxrss;
    return 0;
}

static bool file_is_empty_or_missing(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return true;
    return st.st_size == 0;
}

static std::string csv_escape(const std::string &s) {
    bool need_quote = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') { need_quote = true; break; }
    }
    if (!need_quote) return s;

    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static void append_csv_row(const std::string &csv_path,
                           const std::string &header,
                           const std::string &row)
{
    if (csv_path.empty()) return;

    const bool write_header = file_is_empty_or_missing(csv_path);

    std::ofstream fout(csv_path, std::ios::out | std::ios::app);
    if (!fout.good()) {
        std::cerr << "[WARN] Cannot open CSV for append: " << csv_path << "\n";
        return;
    }
    if (write_header) fout << header << "\n";
    fout << row << "\n";
}


static bool ends_with_ci(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    const std::size_t off = s.size() - suf.size();
    for (std::size_t i = 0; i < suf.size(); ++i) {
        const unsigned char a = static_cast<unsigned char>(s[off + i]);
        const unsigned char b = static_cast<unsigned char>(suf[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

static void print_usage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " graph.bin matroid_factor alpha algo [options] [csv_path]\n\n"
        << "  graph.bin       : tinyGraph .bin (NEW FORMAT includes part_id)\n"
        << "  matroid_factor  : cap[j] = floor(matroid_factor * |P_j|)\n"
        << "  alpha           : swap threshold factor (streaming algos, default 2.0)\n"
        << "  algo            : sg|simple_greedy | fg|fair_greedy | kgis|k_greedy_is\n"
        << "                    | kgts|k_greedy_ts | fksm|fksm_alg\n"
        << "                    | sm|stream_matroid | smw|stream_matroid_w\n"
        << "                    | sgt|stream_greedy_threshold | sr|stream_random\n"
        << "                    | sgr|stream_greedy\n\n"
        << "Options:\n"
        << "  --eps <double>          (FkSM only, default 0.1)\n"
        << "  --seed <uint64>         (FkSM + kGreedyIS + kGreedyTS + StreamRandom, default 42)\n"
        << "  --nonmonotone <0|1>     (default 0)   // set 1 for Max-k-Cut, etc.\n"
        << "  --alpha <double>        (streaming algos swap factor, overrides positional arg)\n"
        << "  csv_path                (optional) append results\n\n"
        << "Notes:\n"
        << "  - Partition matroid is enforced on supp(x) = {u : x[u] != 0}.\n"
        << "  - StreamMatroid  (sm):  one-consistent streaming, exact marginal swaps.\n"
        << "  - StreamMatroidW (smw): one-consistent streaming, weight-cached swaps.\n"
        << "  - StreamGreedyThreshold (sgt): greedy streaming, non-decreasing marginal threshold.\n"
        << "  - StreamRandom (sr):    randomized streaming, 50%% acceptance probability.\n"
        << "  - StreamGreedy (sgr):   greedy streaming, no threshold, accept if matroid-feasible.\n";
}

static matroid::Cap build_cap_from_factor(const mygraph::tinyGraph& g, double matroid_factor) {
    const std::size_t p = matroid::derive_p(g);
    matroid::Cap cap(p, 0);

    std::vector<std::size_t> part_size(p, 0);
    for (std::size_t u = 0; u < g.n; ++u) {
        const std::size_t pid = static_cast<std::size_t>(g.part_id[u]);
        if (pid < p) part_size[pid]++;
    }

    for (std::size_t j = 0; j < p; ++j) {
        const double raw = matroid_factor * static_cast<double>(part_size[j]);
        std::size_t cj = static_cast<std::size_t>(std::floor(raw));
        if (cj > part_size[j]) cj = part_size[j];
        cap[j] = cj;
    }
    return cap;
}

static std::size_t sum_cap_local(const matroid::Cap& cap) {
    std::size_t s = 0;
    for (auto c : cap) s += c;
    return s;
}

enum class AlgoId : int {
    UNKNOWN = 0,
    SIMPLE_GREEDY = 1,
    FAIR_GREEDY = 2,
    K_GREEDY_IS = 3,
    K_GREEDY_TS = 4,
    FKSM = 5,
    STREAM_MATROID = 6,
    STREAM_MATROID_W = 7,
    STREAM_RANDOM = 8,
    STREAM_GREEDY = 9,
};

static AlgoId parse_algo_id(const std::string &algo) {
    if (algo == "sg" || algo == "simple_greedy") return AlgoId::SIMPLE_GREEDY;
    if (algo == "fg" || algo == "fair_greedy")   return AlgoId::FAIR_GREEDY;
    if (algo == "kgis" || algo == "k_greedy_is" || algo == "kgreedyis") return AlgoId::K_GREEDY_IS;
    if (algo == "kgts" || algo == "k_greedy_ts" || algo == "kgreedyts") return AlgoId::K_GREEDY_TS;
    if (algo == "fksm" || algo == "fksm_alg")    return AlgoId::FKSM;
    if (algo == "sm" || algo == "stream_matroid") return AlgoId::STREAM_MATROID;
    if (algo == "smw" || algo == "stream_matroid_w") return AlgoId::STREAM_MATROID_W;
    if (algo == "sr" || algo == "stream_random") return AlgoId::STREAM_RANDOM;
    if (algo == "sgr" || algo == "stream_greedy") return AlgoId::STREAM_GREEDY;
    return AlgoId::UNKNOWN;
}

int main(int argc, char** argv) {
    using namespace std;
    using namespace mygraph;

    if (argc < 5) {
        print_usage(argv[0]);
        return 1;
    }

    string graphFile = argv[1];
    const double matroid_factor = std::stod(argv[2]);
    double alpha = std::stod(argv[3]);
    string algo_str = argv[4];

    double eps = 0.1;                
    std::uint64_t seed = 42ULL;       
    bool is_non_monotone = false;    

    string csv_path;

    for (int i = 5; i < argc; ++i) {
        string tok = argv[i];

        auto need_next = [&](const char* opt) -> const char* {
            if (i + 1 >= argc) {
                cerr << "Error: missing value for " << opt << "\n";
                exit(1);
            }
            return argv[++i];
        };

        if (tok == "--eps") {
            eps = std::strtod(need_next("--eps"), nullptr);
        } else if (tok == "--seed") {
            seed = static_cast<std::uint64_t>(std::strtoull(need_next("--seed"), nullptr, 10));
        } else if (tok == "--nonmonotone") {
            const int v = std::atoi(need_next("--nonmonotone"));
            is_non_monotone = (v != 0);
            ksub::is_non_monotone = is_non_monotone;
        } else if (tok == "--alpha") {
            alpha = std::strtod(need_next("--alpha"), nullptr);
        } else if (!tok.empty() && tok[0] != '-' && ends_with_ci(tok, ".csv")) {
            csv_path = tok;
        } else if (!tok.empty() && tok[0] != '-' && csv_path.empty() && i == argc - 1) {
            csv_path = tok;
        } else {
            cerr << "[WARN] Unknown arg ignored: " << tok << "\n";
        }
    }

    tinyGraph g;
    if (!g.read_binary(graphFile)) {
        cerr << "Error: cannot read binary graph from " << graphFile << "\n";
        return 1;
    }

    cout << "Graph loaded: n = " << g.n
         << ", m = " << g.m
         << ", K = " << g.K
         << ", undirected = " << (g.undirected ? "true" : "false")
         << "\n";
    cout << "algo = " << algo_str << "\n";
    cout << "matroid_factor = " << matroid_factor << "\n";
    cout << "alpha = " << alpha << "\n";
    cout << "is_non_monotone = " << (is_non_monotone ? "true" : "false") << "\n";

    const std::size_t p = matroid::derive_p(g);
    matroid::Cap cap = build_cap_from_factor(g, matroid_factor);
    const std::size_t Bm = sum_cap_local(cap);

    cout << "Partition parts p = " << p << "\n";
    cout << "Matroid rank budget Bm = sum_j cap[j] = " << Bm << "\n";

    // Legacy fairness budget (used by existing algos that still need it)
    const std::size_t B = static_cast<std::size_t>(std::floor(alpha * static_cast<double>(g.n)));

    std::vector<std::size_t> l_limits(g.K + 1, 0), u_limits(g.K + 1, 0);
    if (g.K > 0) {
        const std::size_t li = static_cast<std::size_t>(std::floor(0.8 * (double)B / (double)g.K));
        const std::size_t ui = static_cast<std::size_t>(std::floor(1.4 * (double)B / (double)g.K));
        for (std::size_t i = 1; i <= g.K; ++i) {
            l_limits[i] = li;
            u_limits[i] = ui;
        }
    }

    const long peak_before_kb = getPeakRSS_KB();

    algs::Result res;
    const AlgoId algo_id = parse_algo_id(algo_str);

    switch (algo_id) {
        case AlgoId::SIMPLE_GREEDY:
            res = algs::run_simple_greedy(g, cap);
            break;

        case AlgoId::FAIR_GREEDY: {
            std::vector<std::size_t> u_upper(g.K, 0);
            for (std::size_t i = 1; i <= g.K; ++i) u_upper[i - 1] = u_limits[i];
            res = algs::run_fair_greedy(g, cap, u_upper);
            break;
        }

        case AlgoId::K_GREEDY_IS: {
            std::vector<std::size_t> u_upper(g.K, 0);
            for (std::size_t i = 1; i <= g.K; ++i) u_upper[i - 1] = u_limits[i];
            res = algs::run_k_greedy_is(g, cap, u_upper);
            break;
        }

        case AlgoId::K_GREEDY_TS: {
            std::vector<std::size_t> u_upper(g.K, 0);
            for (std::size_t i = 1; i <= g.K; ++i) u_upper[i - 1] = u_limits[i];
            res = algs::run_k_greedy_ts(g, cap, u_upper);
            break;
        }

        case AlgoId::FKSM:
            res = algs::run_fksm_alg(g, cap, u_limits, l_limits, eps, seed);
            break;

        case AlgoId::STREAM_MATROID:
            res = algs::run_stream_matroid(g, cap, alpha);
            break;

        case AlgoId::STREAM_MATROID_W:
            res = algs::run_stream_matroid_w(g, cap, alpha);
            break;

        case AlgoId::STREAM_RANDOM:
            res = algs::run_stream_random(g, cap, seed);
            break;

        case AlgoId::STREAM_GREEDY:
            res = algs::run_stream_greedy(g, cap);
            break;

        default:
            cerr << "Error: unknown algo = " << algo_str << "\n";
            print_usage(argv[0]);
            return 1;
    }

    const long peak_after_kb = getPeakRSS_KB();
    if (peak_after_kb > peak_before_kb) res.mem_mb = (peak_after_kb - peak_before_kb) / 1024.0;
    else res.mem_mb = 0.0;

    res.matroid_error = static_cast<double>(matroid::matroid_violation(g, res.x, cap));
    res.matroid_checks += 1;

    res.total_error = res.matroid_error;

    cout << fixed << setprecision(6);
    cout << "-----------------------------\n";
    cout << res.algo << " finished.\n";
    cout << "constraint          = " << res.constraint << "\n";
    cout << "f(x)                = " << res.f_value << "\n";
    cout << "MatroidError(x)     = " << res.matroid_error << "\n";
    cout << "TotalError(x)       = " << res.total_error << "\n";
    cout << "#calls (f/marg)     = " << res.queries << "\n";
    cout << "#calls (matroid)    = " << res.matroid_checks << "\n";
    cout << "elapsed time (s)    = " << res.time_sec << "\n";
    cout << "memory used (MB)    = " << res.mem_mb << "\n";

    if (!csv_path.empty()) {
        const string header =
            "algo,constraint,matroid_factor,alpha,Bm,is_non_monotone,eps,seed,"
            "f_value,matroid_error,total_error,queries,matroid_checks,time_sec,mem_mb";

        ostringstream row;
        row << csv_escape(res.algo) << ","
            << csv_escape(res.constraint) << ","
            << setprecision(17) << matroid_factor << ","
            << setprecision(17) << alpha << ","
            << Bm << ","
            << (is_non_monotone ? 1 : 0) << ","
            << setprecision(17) << eps << ","
            << seed << ","
            << setprecision(17) << res.f_value << ","
            << setprecision(17) << res.matroid_error << ","
            << setprecision(17) << res.total_error << ","
            << res.queries << ","
            << res.matroid_checks << ","
            << setprecision(10) << res.time_sec << ","
            << setprecision(6) << res.mem_mb;

        append_csv_row(csv_path, header, row.str());
    }

    return 0;
}
