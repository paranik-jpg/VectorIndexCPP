# VectorIndexCPP

A high-performance, thread-safe, dense vector similarity search engine implemented in modern **C++20**. Designed from the ground up for low-latency Top-K retrieval, contiguous cache-line alignment, hardware prefetching, and lock contention minimization.

---

## Architectural & Mathematical Principles

### 1. Flat Contiguous Memory Hierarchy (Cache Locality)
Instead of fragmented heap allocations (e.g., `std::vector<std::vector<float>>` which incurs pointer chasing and cache misses), VectorIndexCPP stores all vectors in a single contiguous 1D buffer:

```text
vectors_ ∈ ℝ^(N × D)
```

The vector for logical index $i$ begins at offset $i \times D$ and extends to $(i + 1) \times D$. This guarantees:
- Sequential memory streaming that saturates L1/L2 hardware prefetchers.
- Zero pointer indirection overhead.
- Elimination of heap fragmentation.

### 2. Pre-Normalization & Dot Product Equivalence
Cosine similarity between query $\mathbf{q}$ and target $\mathbf{v}$ is defined as:

$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{v}) = \frac{\mathbf{q} \cdot \mathbf{v}}{\|\mathbf{q}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{d=0}^{D-1} q_d v_d}{\sqrt{\sum_{d=0}^{D-1} q_d^2} \sqrt{\sum_{d=0}^{D-1} v_d^2}}$$

By normalizing vectors to unit Euclidean norm ($\|\mathbf{v}\|_2 = 1$) upon insertion and normalizing the query once upon search:

$$\|\mathbf{q}\|_2 = 1, \quad \|\mathbf{v}\|_2 = 1 \implies \text{CosineSimilarity}(\mathbf{q}, \mathbf{v}) = \mathbf{q} \cdot \mathbf{v} = \sum_{d=0}^{D-1} q_d v_d$$

This removes $N$ expensive square root (`sqrtf`) and division operations during query evaluation, reducing the similarity check to pure fused multiply-add sequences.

### 3. Bounded Min-Heap for Top-K Search ($O(N \log K)$)
Rather than sorting all $N$ similarity scores ($O(N \log N)$), VectorIndexCPP maintains a bounded min-heap of size $K$:
- The root of the min-heap always holds the **current $K$-th best score** (the smallest score among the top $K$).
- For each vector $i \in [0, N-1]$, if the heap size is $< K$, the element is pushed ($O(\log K)$).
- Once the heap has $K$ elements, any candidate with $\text{score} \le \text{root.score}$ is discarded in $O(1)$ time.
- If $\text{score} > \text{root.score}$, the root is popped and the candidate inserted in $O(\log K)$ time.
- Overall search complexity: $O(N \log K)$ time and $O(K)$ auxiliary memory.

### 4. Reader-Writer Concurrency (`std::shared_mutex`)
- **Shared Access (Read)**: Multiple search threads acquire `std::shared_lock<std::shared_mutex>` concurrently without blocking each other.
- **Exclusive Access (Write)**: Ingestion threads acquire `std::unique_lock<std::shared_mutex>` when appending new vectors to guarantee memory safety during potential reallocations.
- **Critical Section Minimization**: Dimension validation and query pre-normalization occur **before** acquiring the reader lock, keeping lock hold time bounded strictly to the dot-product scan.

---

## Project Structure

```text
.
├── CMakeLists.txt              # Build configuration (C++20 standard required)
├── LICENSE                     # MIT License
├── README.md                   # Project documentation and architectural guide
├── include/
│   └── vector_index.hpp        # Core VectorIndexCPP implementation
└── tests/
    ├── test_step1.cpp          # Flat memory layout & Rule of Five verification
    ├── test_step2.cpp          # L2 normalization & dot product accuracy tests
    ├── test_step3.cpp          # Bounded min-heap Top-K retrieval tests
    ├── test_step4.cpp          # Multi-threaded read/write concurrency tests
    └── test_step5.cpp          # Dimension safety, micro-benchmarks & TSan tests
```

---

## Building and Running

### Prerequisites
- Clang (>= 13) or GCC (>= 11) supporting C++20
- CMake (>= 3.20)
- Ninja or Make

### 1. Build
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### 2. Run Tests
```bash
./test_step1
./test_step2
./test_step3
./test_step4
./test_step5
```

### 3. Running with ThreadSanitizer (TSan)
To verify race condition freedom under high concurrency:
```bash
mkdir -p build_tsan && cd build_tsan
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
cmake --build .
./test_step5
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
