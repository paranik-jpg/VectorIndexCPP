#include "vector_index.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <thread>
#include <atomic>

// Helper to generate a random vector with a uniform distribution
std::vector<float> generate_random_vector(size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> vec(dim);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
    return vec;
}

// -------------------------------------------------------------
// Test 1: Dimension Guard Verification
// -------------------------------------------------------------
void test_dimension_guards() {
    std::cout << "[Test 1] Running Dimension Guard Verification..." << std::endl;

    const size_t correct_dim = 384;
    const size_t mismatched_dim = 512;
    vector_index::VectorIndexCPP index(correct_dim);

    std::mt19937 rng(42);
    auto valid_vec = generate_random_vector(correct_dim, rng);
    auto invalid_vec = generate_random_vector(mismatched_dim, rng);

    // 1. Ingestion dimension mismatch (512 into 384)
    bool add_exception_thrown = false;
    try {
        index.add(1, invalid_vec);
    } catch (const std::invalid_argument& ex) {
        add_exception_thrown = true;
    }
    assert(add_exception_thrown);
    assert(index.size() == 0); // Index must remain untouched

    // Ingest one valid vector to test query guards
    index.add(100, valid_vec);
    assert(index.size() == 1);

    // 2. Query dimension mismatch (512 query on 384 index)
    bool query_exception_thrown = false;
    try {
        // This suppresses the unused-parameter warning without removing the parameter.
        [[maybe_unused]] auto res = index.search_top_k(invalid_vec, 5);
    } catch (const std::invalid_argument& ex) {
        query_exception_thrown = true;
    }
    assert(query_exception_thrown);

    // 3. Valid query should succeed without issue
    auto results = index.search_top_k(valid_vec, 5);
    assert(results.size() == 1);
    assert(results[0].id == 100);

    std::cout << "  -> Passed: Clean exceptions thrown on dimension mismatch, memory intact.\n" << std::endl;
}

// -------------------------------------------------------------
// Test 2: Micro-Benchmark (p50 and p99 Latency on 10,000 384-D vectors)
// -------------------------------------------------------------
void test_micro_benchmark() {
    std::cout << "[Test 2] Running Micro-Benchmark on 10,000 vectors of 384 dimensions..." << std::endl;

    const size_t N = 10000;
    const size_t D = 384;
    const size_t K = 5;
    // In TSan mode, run 50 queries to keep instrumented test fast; otherwise run 1000
// If compiled with -fsanitize=thread, the compiler turns this flag on,
// directing the preprocessor to compile the block immediately following.
#ifdef __SANITIZE_THREAD__
    const size_t NUM_QUERIES = 50;
#else
    // Evaluated when __SANITIZE_THREAD__ is not defined
    const size_t NUM_QUERIES = 1000;
#endif

    vector_index::VectorIndexCPP index(D);
    std::mt19937 rng(1337);

    // Ingestion phase
    auto start_ingest = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        auto vec = generate_random_vector(D, rng);
        index.add(static_cast<uint64_t>(i + 1), vec);
    }
    auto end_ingest = std::chrono::high_resolution_clock::now();
    auto ingest_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ingest - start_ingest).count();
    std::cout << "  Ingested " << N << " vectors in " << ingest_duration << " ms ("
              << (N * 1000.0 / ingest_duration) << " vectors/sec)" << std::endl;

    // Generate test queries
    std::vector<std::vector<float>> queries;
    queries.reserve(NUM_QUERIES);
    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        queries.push_back(generate_random_vector(D, rng));
    }

    // Benchmark Top-K latency
    std::vector<double> latencies_us;
    latencies_us.reserve(NUM_QUERIES);

    for (size_t i = 0; i < NUM_QUERIES; ++i) {
        auto q_start = std::chrono::high_resolution_clock::now();
        auto results = index.search_top_k(queries[i], K);
        auto q_end = std::chrono::high_resolution_clock::now();

        assert(results.size() == K);

        double elapsed_us = std::chrono::duration<double, std::micro>(q_end - q_start).count();
        latencies_us.push_back(elapsed_us);
    }

    // Sort latencies to compute percentiles
    std::sort(latencies_us.begin(), latencies_us.end());

    double p50 = latencies_us[static_cast<size_t>(NUM_QUERIES * 0.50)];
    double p90 = latencies_us[static_cast<size_t>(NUM_QUERIES * 0.90)];
    double p99 = latencies_us[static_cast<size_t>(NUM_QUERIES * 0.99)];
    double min_lat = latencies_us.front();
    double max_lat = latencies_us.back();

    double total_us = 0.0;
    for (double lat : latencies_us) {
        total_us += lat;
    }
    double avg_lat = total_us / NUM_QUERIES;
    double qps = (NUM_QUERIES * 1000000.0) / total_us;

    std::cout << "  === Latency Benchmark Results (Top-" << K << " over " << N << " vectors, " << D << "-D) ===" << std::endl;
    std::cout << "    Min Latency: " << min_lat << " us (" << min_lat / 1000.0 << " ms)" << std::endl;
    std::cout << "    Avg Latency: " << avg_lat << " us (" << avg_lat / 1000.0 << " ms)" << std::endl;
    std::cout << "    p50 Latency: " << p50 << " us (" << p50 / 1000.0 << " ms)" << std::endl;
    std::cout << "    p90 Latency: " << p90 << " us (" << p90 / 1000.0 << " ms)" << std::endl;
    std::cout << "    p99 Latency: " << p99 << " us (" << p99 / 1000.0 << " ms)" << std::endl;
    std::cout << "    Max Latency: " << max_lat << " us (" << max_lat / 1000.0 << " ms)" << std::endl;
    std::cout << "    Throughput:  " << static_cast<size_t>(qps) << " queries/sec (QPS)\n" << std::endl;
}

// -------------------------------------------------------------
// Test 3: Thread Safety Under High Concurrency
// -------------------------------------------------------------
void test_tsan_concurrency() {
    std::cout << "[Test 3] Running High Concurrency Thread-Safety Verification..." << std::endl;

    const size_t D = 64;
    vector_index::VectorIndexCPP index(D);
    std::mt19937 rng(999);

    // Initial seed
    for (size_t i = 0; i < 50; ++i) {
        auto vec = generate_random_vector(D, rng);
        index.add(static_cast<uint64_t>(i + 1), vec);
    }

    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> search_count{0};
    std::atomic<uint64_t> insert_count{0};

    // 4 concurrent reader threads
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&index, &stop_flag, &search_count, D, t]() {
            std::mt19937 local_rng(100 + t);
            while (!stop_flag.load(std::memory_order_relaxed)) {
                auto query = generate_random_vector(D, local_rng);
                auto res = index.search_top_k(query, 5);
                assert(!res.empty());
                search_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // 1 concurrent writer thread constantly inserting new vectors
    std::thread writer([&index, &stop_flag, &insert_count, D]() {
        std::mt19937 writer_rng(2026);
        uint64_t id = 10000;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            auto vec = generate_random_vector(D, writer_rng);
            index.add(id++, vec);
            insert_count.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::yield();
        }
    });

    // Run concurrency test for 300 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop_flag.store(true, std::memory_order_relaxed);

    for (auto& r : readers) {
        r.join();
    }
    writer.join();

    std::cout << "  -> High Concurrency Success: "
              << search_count.load() << " searches and "
              << insert_count.load() << " writes safely executed.\n" << std::endl;
}

int main() {
    test_dimension_guards();
    test_micro_benchmark();
    test_tsan_concurrency();
    std::cout << "All Step 5 tests and benchmarks completed successfully!" << std::endl;
    return 0;
}
