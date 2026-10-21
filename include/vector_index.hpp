#pragma once // A preprocessor directive ensuring this header file is included only once per compilation unit

#include <cstddef>   // For std::size_t
#include <cstdint>   // For std::uint64_t
#include <vector>
#include <span>
#include <stdexcept> // For standard exception types
#include <string>

// Enclose all index types inside a dedicated namespace to avoid name collisions.
namespace vector_index {

/**
 * @brief Thread-safe dense vector index interface and contiguous memory storage.
 *
 * Implements a flat vector buffer optimized for CPU cache locality:
 * - Single flat buffer std::vector<float> stores all dimensions contiguously.
 * - Entity at logical index i occupies [i * D, (i + 1) * D).
 * - Parallel buffer std::vector<uint64_t> maps logical index i back to client ID.
 * - Copy operations are deleted to prevent massive, accidental deep copies.
 */
class VectorIndexCPP {
public:
    // Explicit constructor requiring the vector dimension D
    // Explicit => We can't do VectorIndexCPP index = 128;   // ERROR
    explicit VectorIndexCPP(size_t dimensions)
        : dimension_(dimensions) {
        if (dimensions == 0) {
            // throw => Stops the normal execution of the operation and reports an error to the caller.
            throw std::invalid_argument("Vector dimension must be strictly greater than 0.");
        }
    }

    // Virtual destructor for safe polymorphic inheritance/destruction
    virtual ~VectorIndexCPP() = default;

    // Rule of Five: Delete copy constructor and copy assignment operator
    // to prevent accidental deep-copying of massive vector buffers.
    VectorIndexCPP(const VectorIndexCPP&) = delete;                 // We can't do VectorIndexCPP b(a).
    VectorIndexCPP& operator=(const VectorIndexCPP&) = delete;      // We can't do b = a.

    // Allow default move operations
    VectorIndexCPP(VectorIndexCPP&&) noexcept = default;            // We can do VectorIndexCPP b(std::move(a)).
    VectorIndexCPP& operator=(VectorIndexCPP&&) noexcept = default; // We can do b = std::move(a).

    // [[nodiscard]] => Warns the caller if they call the method but ignore the return value.
    // noexcept => Guarantees it will never throw.
    // Accessor for the configured vector dimensionality
    [[nodiscard]] size_t dimension() const noexcept {
        return dimension_;
    }

    // Returns current number of indexed vectors
    [[nodiscard]] size_t size() const noexcept {
        return ids_.size();
    }

    // Checks if the index contains zero vectors
    [[nodiscard]] bool empty() const noexcept {
        return ids_.empty();
    }

    // Obtain a read-only C++20 span view of the contiguous vector at logical index i
    [[nodiscard]] std::span<const float> get_vector(size_t index) const {
        if (index >= size()) {
            throw std::out_of_range("Vector index " + std::to_string(index) +
                                    " out of range (size: " + std::to_string(size()) + ")");
        }
        return std::span<const float>(vectors_.data() + (index * dimension_), dimension_);
    }

    // Retrieve client-facing identifier associated with logical index i
    [[nodiscard]] uint64_t get_id(size_t index) const {
        if (index >= size()) {
            throw std::out_of_range("ID index " + std::to_string(index) +
                                    " out of range (size: " + std::to_string(size()) + ")");
        }
        return ids_[index];
    }

    // Direct read-only pointer to the underlying continuous float array for high-speed linear scans / SIMD
    [[nodiscard]] const float* data() const noexcept {
        return vectors_.data();
    }

    // Direct read-only pointer to the underlying IDs buffer
    [[nodiscard]] const uint64_t* id_data() const noexcept {
        return ids_.data();
    }

// Protected so derived index implementations can read and mutate these buffers.
protected:
    size_t dimension_;                   // Fixed dimension D for all vectors in this index
    std::vector<float> vectors_;         // Flat continuous buffer: size is N * D
    std::vector<uint64_t> ids_;          // Parallel identifier buffer: size is N
};

} // namespace vector_index
