#include "vector_index.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

void test_top_k_search() {
    const size_t D = 3;
    vector_index::VectorIndexCPP index(D);

    // Ingest 5 vectors with known directions:
    // Vector 101: perfectly aligned with X-axis [1, 0, 0]
    index.add(101, std::vector<float>{1.0f, 0.0f, 0.0f});
    // Vector 102: 45 degrees between X and Y [1, 1, 0] -> normalized [0.7071, 0.7071, 0]
    index.add(102, std::vector<float>{1.0f, 1.0f, 0.0f});
    // Vector 103: aligned with Y-axis [0, 1, 0]
    index.add(103, std::vector<float>{0.0f, 1.0f, 0.0f});
    // Vector 104: aligned with Z-axis [0, 0, 1]
    index.add(104, std::vector<float>{0.0f, 0.0f, 1.0f});
    // Vector 105: negative X-axis [-1, 0, 0] (opposite direction)
    index.add(105, std::vector<float>{-1.0f, 0.0f, 0.0f});

    assert(index.size() == 5);

    // Query along X-axis: [2.0, 0.0, 0.0] (should normalize to [1, 0, 0])
    std::vector<float> query = {2.0f, 0.0f, 0.0f};

    // Ask for Top 3
    auto top3 = index.search_top_k(query, 3);
    assert(top3.size() == 3);

    // Expected order:
    // 1st: ID 101 with dot product = 1.0
    // 2nd: ID 102 with dot product = ~0.707106
    // 3rd: ID 103 or 104 with dot product = 0.0
    assert(top3[0].id == 101);
    assert(std::abs(top3[0].score - 1.0f) < 1e-4f);

    assert(top3[1].id == 102);
    assert(std::abs(top3[1].score - (1.0f / std::sqrt(2.0f))) < 1e-4f);

    // 3rd must have similarity 0.0
    assert(std::abs(top3[2].score - 0.0f) < 1e-4f);

    // Verify descending order
    assert(top3[0].score >= top3[1].score);
    assert(top3[1].score >= top3[2].score);

    // Ask for K larger than total index size (e.g. K = 10, total = 5)
    auto top10 = index.search_top_k(query, 10);
    assert(top10.size() == 5);
    assert(top10.front().id == 101);
    assert(top10.back().id == 105);
    assert(std::abs(top10.back().score - (-1.0f)) < 1e-4f);

    // Ask for K = 0
    auto top0 = index.search_top_k(query, 0);
    assert(top0.empty());

    // Query dimension mismatch test
    std::vector<float> bad_query = {1.0f, 0.0f};
    bool caught = false;
    try {
        [[maybe_unused]] auto res = index.search_top_k(bad_query, 3);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
}

int main() {
    test_top_k_search();
    std::cout << "All Step 3 tests passed successfully!" << std::endl;
    return 0;
}
