#include "vector_index.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_normalization_routine() {
    // Vector: [3.0, 4.0] -> L2 Norm is sqrt(3^2 + 4^2) = 5.0
    // Normalized: [3/5, 4/5] = [0.6, 0.8]
    std::vector<float> v = {3.0f, 4.0f};
    float original_norm = vector_index::VectorIndexCPP::normalize_l2(v);

    assert(std::abs(original_norm - 5.0f) < 1e-5f);
    assert(std::abs(v[0] - 0.6f) < 1e-5f);
    assert(std::abs(v[1] - 0.8f) < 1e-5f);

    // After normalization, new L2 norm must be 1.0
    float new_norm = std::sqrt(v[0] * v[0] + v[1] * v[1]);
    assert(std::abs(new_norm - 1.0f) < 1e-5f);

    // Zero vector test: should not divide by zero or crash
    std::vector<float> zero_vec = {0.0f, 0.0f, 0.0f};
    float zero_norm = vector_index::VectorIndexCPP::normalize_l2(zero_vec);
    assert(zero_norm == 0.0f);
    assert(zero_vec[0] == 0.0f && zero_vec[1] == 0.0f && zero_vec[2] == 0.0f);
}

void test_dot_product() {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    float dot = vector_index::VectorIndexCPP::dot_product(a.data(), b.data(), 3);
    assert(std::abs(dot - 32.0f) < 1e-5f);
}

void test_ingestion_and_cosine_equivalence() {
    const size_t D = 3;
    vector_index::VectorIndexCPP index(D);

    // Insert vector 1001: [3.0, 0.0, 4.0] -> norm 5.0 -> stored as [0.6, 0.0, 0.8]
    std::vector<float> raw_v1 = {3.0f, 0.0f, 4.0f};
    size_t idx0 = index.add(1001, raw_v1);
    assert(idx0 == 0);

    // Insert vector 1002: [0.0, 5.0, 0.0] -> norm 5.0 -> stored as [0.0, 1.0, 0.0]
    std::vector<float> raw_v2 = {0.0f, 5.0f, 0.0f};
    size_t idx1 = index.add(1002, raw_v2);
    assert(idx1 == 1);

    // Verify stored vectors are unit length
    auto slice0 = index.get_vector(0);
    assert(std::abs(slice0[0] - 0.6f) < 1e-5f);
    assert(std::abs(slice0[1] - 0.0f) < 1e-5f);
    assert(std::abs(slice0[2] - 0.8f) < 1e-5f);

    auto slice1 = index.get_vector(1);
    assert(std::abs(slice1[0] - 0.0f) < 1e-5f);
    assert(std::abs(slice1[1] - 1.0f) < 1e-5f);
    assert(std::abs(slice1[2] - 0.0f) < 1e-5f);

    // Query with a vector identical in direction to vector 0: [6.0, 0.0, 8.0]
    std::vector<float> query = {6.0f, 0.0f, 8.0f};
    vector_index::VectorIndexCPP::normalize_l2(query); // query becomes [0.6, 0.0, 0.8]

    // Dot product with vector 0 should be exactly 1.0 (perfect match in angle)
    float sim0 = vector_index::VectorIndexCPP::dot_product(query.data(), slice0.data(), D);
    assert(std::abs(sim0 - 1.0f) < 1e-5f);

    // Dot product with orthogonal vector 1 should be 0.0
    float sim1 = vector_index::VectorIndexCPP::dot_product(query.data(), slice1.data(), D);
    assert(std::abs(sim1 - 0.0f) < 1e-5f);

    // Check dimension mismatch throws exception
    std::vector<float> wrong_dim = {1.0f, 2.0f};
    bool caught = false;
    try {
        index.add(9999, wrong_dim);
    } catch (const std::invalid_argument&) {
        caught = true;
    }
    assert(caught);
}

int main() {
    test_normalization_routine();
    test_dot_product();
    test_ingestion_and_cosine_equivalence();
    std::cout << "All Step 2 tests passed successfully!" << std::endl;
    return 0;
}

