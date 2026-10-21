#include "vector_index.hpp"
#include <iostream>
#include <cassert>
#include <type_traits> // For the first arguments of 4 static_asserts !

// Derived helper for testing and validating the interface in step 1
class TestIndex : public vector_index::VectorIndexCPP {
public:
    explicit TestIndex(size_t dim) : VectorIndexCPP(dim) {}

    void insert_mock(uint64_t id, const std::vector<float>& vec) {
        assert(vec.size() == dimension_);
        ids_.push_back(id);
        vectors_.insert(vectors_.end(), vec.begin(), vec.end());
    }
};

int main() {
    // static_assert(condition, message) evaluates condition at compile time.
    // 1. Verify Deleted Copy Constructor and Deleted Copy Assignment at compile-time
    static_assert(!std::is_copy_constructible_v<vector_index::VectorIndexCPP>,
                  "VectorIndexCPP must not be copy-constructible!");
    static_assert(!std::is_copy_assignable_v<vector_index::VectorIndexCPP>,
                  "VectorIndexCPP must not be copy-assignable!");

    // 2. Verify Move operations are enabled
    static_assert(std::is_nothrow_move_constructible_v<vector_index::VectorIndexCPP>,
                  "VectorIndexCPP should be noexcept move-constructible!");
    static_assert(std::is_nothrow_move_assignable_v<vector_index::VectorIndexCPP>,
                  "VectorIndexCPP should be noexcept move-assignable!");

    const size_t D = 3;
    TestIndex index(D);

    // If condition is false at runtime => aborts the program immediately and prints the exact line that failed.
    assert(index.dimension() == 3);
    assert(index.empty());
    assert(index.size() == 0);

    // Insert 2 mock vectors
    // Vector 0: ID 1001, values [1.0, 2.0, 3.0]
    index.insert_mock(1001, {1.0f, 2.0f, 3.0f});
    // Vector 1: ID 1002, values [4.0, 5.0, 6.0]
    index.insert_mock(1002, {4.0f, 5.0f, 6.0f});

    assert(index.size() == 2);
    assert(!index.empty());

    // Verify IDs
    assert(index.get_id(0) == 1001);
    assert(index.get_id(1) == 1002);

    // Verify continuous slices [i * D, (i + 1) * D)
    std::span<const float> slice0 = index.get_vector(0);
    assert(slice0.size() == 3);
    assert(slice0[0] == 1.0f && slice0[1] == 2.0f && slice0[2] == 3.0f);

    std::span<const float> slice1 = index.get_vector(1);
    assert(slice1.size() == 3);
    assert(slice1[0] == 4.0f && slice1[1] == 5.0f && slice1[2] == 6.0f);

    // Verify underlying continuous memory
    const float* raw_data = index.data();
    assert(raw_data[0] == 1.0f && raw_data[1] == 2.0f && raw_data[2] == 3.0f &&
           raw_data[3] == 4.0f && raw_data[4] == 5.0f && raw_data[5] == 6.0f);

    std::cout << "All Step 1 tests passed successfully!" << std::endl;
    return 0;
}
