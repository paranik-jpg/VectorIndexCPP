#include "vector_index.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

void test_concurrent_readers_and_writers() {
    const size_t D = 8;
    vector_index::VectorIndexCPP index(D);

    // Initial seed: 10 vectors
    for (uint64_t i = 0; i < 10; ++i) {
        std::vector<float> v(D, static_cast<float>(i + 1));
        index.add(i, v);
    }

    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> total_searches{0};
    std::atomic<uint64_t> total_inserts{0};

    // Launch 4 concurrent reader threads performing continuous similarity searches
    std::vector<std::thread> reader_threads;
    for (int t = 0; t < 4; ++t) {
        reader_threads.emplace_back([&index, &stop_flag, &total_searches, D]() {
            std::vector<float> query(D, 1.0f);
            // std::memory_order_relaxed avoids heavy CPU memory barrier instructions
            // since strict ordering isn't required for a simple stop signal.
            while (!stop_flag.load(std::memory_order_relaxed)) {
                auto results = index.search_top_k(query, 5);
                assert(!results.empty());
                // Verify descending order
                for (size_t i = 1; i < results.size(); ++i) {
                    assert(results[i - 1].score >= results[i].score);
                }
                // Atomically increment our total search counter.
                total_searches.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Launch 2 concurrent writer threads inserting new vectors
    std::vector<std::thread> writer_threads;
    for (int t = 0; t < 2; ++t) {
        writer_threads.emplace_back([&index, &stop_flag, &total_inserts, D, t]() {
            uint64_t local_id = 1000 + (t * 10000);
            while (!stop_flag.load(std::memory_order_relaxed)) {
                std::vector<float> v(D, static_cast<float>(local_id % 50 + 1));
                index.add(local_id++, v);
                // Atomically increment the count of successful insertions.
                total_inserts.fetch_add(1, std::memory_order_relaxed);
                // yield() allows the current thread to yield execution,
                // enabling the scheduler to switch to another thread if one is ready to run.
                std::this_thread::yield();
            }
        });
    }

    // Let threads run concurrently for 200 milliseconds
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    stop_flag.store(true, std::memory_order_relaxed);

    // Join all threads
    for (auto& th : reader_threads) {
        th.join();
    }
    for (auto& th : writer_threads) {
        th.join();
    }

    std::cout << "Concurrent stress test passed!\n"
              << "  Total successful parallel searches: " << total_searches.load() << "\n"
              << "  Total successful parallel inserts:  " << total_inserts.load() << "\n"
              << "  Final index size:                   " << index.size() << std::endl;

    assert(index.size() == 10 + total_inserts.load());
}

void test_dimension_validation_before_lock() {
    vector_index::VectorIndexCPP index(4);

    // Mismatched insertion dimensions
    std::vector<float> bad_add = {1.0f, 2.0f};
    bool caught_add = false;
    try {
        index.add(99, bad_add);
    } catch (const std::invalid_argument&) {
        caught_add = true;
    }
    assert(caught_add);

    // Mismatched query dimensions
    std::vector<float> bad_query = {1.0f, 2.0f, 3.0f};
    bool caught_query = false;
    try {
        [[maybe_unused]] auto res = index.search_top_k(bad_query, 5);
    } catch (const std::invalid_argument&) {
        caught_query = true;
    }
    assert(caught_query);
}

int main() {
    test_dimension_validation_before_lock();
    test_concurrent_readers_and_writers();
    std::cout << "All Step 4 tests passed successfully!" << std::endl;
    return 0;
}
