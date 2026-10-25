#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <stdexcept>
#include <string>
#include <cmath>
#include <numeric>
#include <queue>
#include <algorithm>
#include <shared_mutex>
#include <mutex>

namespace vector_index {

/**
 * @brief Search result structure holding client ID and similarity score.
 */
struct SearchResult {
    uint64_t id;
    float score;

    // Operator overloading!
    // Strict equality comparison for unit tests / assertions
    bool operator==(const SearchResult& other) const noexcept {
        return id == other.id && std::abs(score - other.score) < 1e-5f;
    }
};

/**
 * @brief Comparator for bounded Min-Heap:
 * std::priority_queue uses operator() as a "less" predicate.
 * When operator() returns (a.score > b.score), smaller scores are placed
 * at the top of the heap, making it a Min-Heap.
 */
struct MinHeapComparator {
    bool operator()(const SearchResult& a, const SearchResult& b) const noexcept {
        return a.score > b.score;
    }
};

/**
 * @brief Thread-safe dense vector index interface and contiguous memory storage.
 *
 * Implements a flat vector buffer optimized for CPU cache locality and thread safety:
 * - Single flat buffer std::vector<float> stores all dimensions contiguously.
 * - Entity at logical index i occupies [i * D, (i + 1) * D).
 * - Parallel buffer std::vector<uint64_t> maps logical index i back to client ID.
 * - Ingestion pipeline normalizes all vectors to unit L2 length (||v||2 = 1.0).
 * - Inner similarity collapses directly to pure dot product, eliminating runtime sqrt.
 * - Bounded min-heap extracts Top-K nearest neighbors in O(N log K) time.
 * - Concurrency control:
 *     * mutable std::shared_mutex protects internal flat buffers.
 *     * std::shared_lock allows multiple concurrent reader search queries.
 *     * std::unique_lock guarantees exclusive write access during vector ingestion.
 *     * Input validations are performed prior to acquiring locks.
 * - Copy operations are deleted to prevent massive, accidental deep copies.
 */
class VectorIndexCPP {
protected:
    size_t dimension_;                          // Fixed dimension D for all vectors in this index
    std::vector<float> vectors_;                 // Flat continuous buffer: size is N * D
    std::vector<uint64_t> ids_;                  // Parallel identifier buffer: size is N
    mutable std::shared_mutex rw_mutex_;         // Reader-writer mutex protecting vectors_ and ids_

public:
    // Explicit constructor requiring the vector dimension D
    explicit VectorIndexCPP(size_t dimensions)
        : dimension_(dimensions) {
        if (dimensions == 0) {
            throw std::invalid_argument("Vector dimension must be strictly greater than 0.");
        }
    }

    // Virtual destructor for safe polymorphic inheritance/destruction
    virtual ~VectorIndexCPP() = default;

    // Rule of Five: Delete copy constructor and copy assignment operator
    // to prevent accidental deep-copying of massive vector buffers.
    VectorIndexCPP(const VectorIndexCPP&) = delete;
    VectorIndexCPP& operator=(const VectorIndexCPP&) = delete;

    // Move constructor & assignment
    VectorIndexCPP(VectorIndexCPP&& other) noexcept {
        // Safely locks other before moving its vectors,
        // ensuring no reader or writer is actively modifying other during a move.
        std::unique_lock<std::shared_mutex> lock(other.rw_mutex_);
        dimension_ = other.dimension_;
        vectors_ = std::move(other.vectors_);
        ids_ = std::move(other.ids_);
    }

    VectorIndexCPP& operator=(VectorIndexCPP&& other) noexcept {
        if (this != &other) {
            // Lock both mutexes without deadlock using std::lock
            std::unique_lock<std::shared_mutex> lock_this(rw_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock_other(other.rw_mutex_, std::defer_lock);
            std::lock(lock_this, lock_other);

            dimension_ = other.dimension_;
            vectors_ = std::move(other.vectors_);
            ids_ = std::move(other.ids_);
        }
        return *this;
    }

    // Accessor for the configured vector dimensionality (immutable, lock-free)
    [[nodiscard]] size_t dimension() const noexcept {
        return dimension_;
    }

    // Returns current number of indexed vectors (thread-safe shared read)
    [[nodiscard]] size_t size() const noexcept {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        return ids_.size();
    }

    // Checks if the index contains zero vectors (thread-safe shared read)
    [[nodiscard]] bool empty() const noexcept {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        return ids_.empty();
    }

    // Retrieve client-facing identifier associated with logical index i (thread-safe shared read)
    [[nodiscard]] uint64_t get_id(size_t index) const {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        if (index >= ids_.size()) {
            throw std::out_of_range("ID index " + std::to_string(index) +
                                    " out of range (size: " + std::to_string(ids_.size()) + ")");
        }
        return ids_[index];
    }

    // Obtain a read-only C++20 span view of the contiguous vector at logical index i (thread-safe shared read)
    [[nodiscard]] std::span<const float> get_vector(size_t index) const {
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);
        if (index >= ids_.size()) {
            throw std::out_of_range("Vector index " + std::to_string(index) +
                                    " out of range (size: " + std::to_string(ids_.size()) + ")");
        }
        return std::span<const float>(vectors_.data() + (index * dimension_), dimension_);
    }

    // Direct read-only pointer to the underlying continuous float array
    [[nodiscard]] const float* data() const noexcept {
        return vectors_.data();
    }

    // Direct read-only pointer to the underlying IDs buffer
    [[nodiscard]] const uint64_t* id_data() const noexcept {
        return ids_.data();
    }

    /**
     * @brief In-place unit L2 normalization routine.
     * Computes the Euclidean norm ||v||2 = sqrt(sum(v_j^2)) and divides every element by it.
     * If the norm is 0 (or close to 0), elements remain 0 to prevent division by zero.
     *
     * @param vec A mutable span of float coordinates.
     * @return float The original L2 norm before normalization.
     */
    static float normalize_l2(std::span<float> vec) noexcept {
        float sum_sq = 0.0f;
        for (float val : vec) {
            sum_sq += val * val;
        }

        const float norm = std::sqrt(sum_sq);
        constexpr float epsilon = 1e-12f;

        if (norm > epsilon) {
            const float inv_norm = 1.0f / norm;
            for (float& val : vec) {
                val *= inv_norm;
            }
        }
        return norm;
    }

    /**
     * @brief Computes raw dot product between two contiguous memory buffers.
     * When vectors are pre-normalized to unit length, this equals Cosine Similarity.
     *
     * @param a Pointer to first contiguous float array.
     * @param b Pointer to second contiguous float array.
     * @param dim Number of dimensions to compute.
     * @return float Dot product result sum(a_j * b_j).
     */
    [[nodiscard]] static float dot_product(const float* a, const float* b, size_t dim) noexcept {
        float dot = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    /**
     * @brief Ingestion path: adds a vector into the index with thread-safe exclusive lock.
     *
     * 1. Validates dimensions BEFORE acquiring any lock.
     * 2. Acquires exclusive std::unique_lock to prevent data corruption/reallocation races.
     * 3. Appends ID and normalized coordinates to flat buffers.
     *
     * @param id Client-facing unique identifier.
     * @param vec Non-owning span of the vector coordinates to ingest.
     * @return size_t Logical index assigned to this vector.
     */
    size_t add(uint64_t id, std::span<const float> vec) {
        // Validation step 1: Validate dimensions before locking or modifying buffers
        if (vec.size() != dimension_) {
            throw std::invalid_argument("Input vector dimension mismatch: expected " +
                                        std::to_string(dimension_) + ", received " +
                                        std::to_string(vec.size()));
        }

        // Pre-normalize into a local buffer so we minimize time spent holding the exclusive lock
        std::vector<float> normalized_coords(vec.begin(), vec.end());
        normalize_l2(normalized_coords);

        // Exclusive write lock: guarantees single-writer safety during buffer reallocations
        std::unique_lock<std::shared_mutex> lock(rw_mutex_);

        const size_t logical_index = ids_.size();
        ids_.push_back(id);

        vectors_.insert(vectors_.end(), normalized_coords.begin(), normalized_coords.end());

        return logical_index;
    }

    /**
     * @brief Performs Top-K similarity search using reader-writer shared locking and a bounded min-heap.
     *
     * 1. Validates query dimensions BEFORE acquiring any lock.
     * 2. Normalizes local query copy before locking to minimize lock contention.
     * 3. Acquires std::shared_lock allowing multiple readers to search concurrently.
     * 4. Scans contiguous memory slice-by-slice, computing dot product.
     * 5. Uses a bounded min-heap to retain Top-K candidates.
     * 6. Returns results sorted in descending order of similarity.
     *
     * @param query Non-owning view of query vector coordinates.
     * @param k Maximum number of top nearest neighbors to return.
     * @return std::vector<SearchResult> Sorted in descending order of similarity.
     */
    [[nodiscard]] std::vector<SearchResult> search_top_k(std::span<const float> query, size_t k) const {
        // Validation step 1: Validate dimensions before locking
        if (query.size() != dimension_) {
            throw std::invalid_argument("Query vector dimension mismatch: expected " +
                                        std::to_string(dimension_) + ", received " +
                                        std::to_string(query.size()));
        }

        if (k == 0) {
            return {};
        }

        // Pre-normalize local query copy OUTSIDE the lock to reduce lock holding duration
        std::vector<float> normalized_query(query.begin(), query.end());
        normalize_l2(normalized_query);

        // Acquire shared read lock: multiple readers search concurrently without blocking
        std::shared_lock<std::shared_mutex> lock(rw_mutex_);

        const size_t n = ids_.size();
        if (n == 0) {
            return {};
        }

        // Bounded min-heap: top element is always the candidate with the SMALLEST similarity
        std::priority_queue<SearchResult, std::vector<SearchResult>, MinHeapComparator> min_heap;

        const float* query_ptr = normalized_query.data();
        const float* flat_vectors = vectors_.data();

        for (size_t i = 0; i < n; ++i) {
            const float* candidate_ptr = flat_vectors + (i * dimension_);
            const float similarity = dot_product(query_ptr, candidate_ptr, dimension_);
            const uint64_t candidate_id = ids_[i];

            if (min_heap.size() < k) {
                min_heap.push({candidate_id, similarity});
            } else if (similarity > min_heap.top().score) {
                min_heap.pop();
                min_heap.push({candidate_id, similarity});
            }
        }

        // Drain min-heap into result vector in descending order (highest score first)
        std::vector<SearchResult> results(min_heap.size());
        for (size_t i = results.size(); i-- > 0; ) {
            results[i] = min_heap.top();
            min_heap.pop();
        }

        return results;
    }

};

} // namespace vector_index
