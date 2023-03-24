# Hardware Acceleration

*DATE: 2022/11/ 22*

---

Layers in machine learning frameworks

- ML Models
- Computational graph
- Tensor linear algebra libraries
- Different hardware environments

**General acceleration techniques**

- Vectorization

  Modern acceleration instruction allows us to load a contiguous piece of memory - **vector registers** to run arithmetic.

  ```c++
  void vecadd(float* A, float *B, float* C) {
      // Adding two arrays of length 256
      for (int i = 0; i < 64; ++i) {
          float4 a = load_float4(A + i*4);
          float4 b = load_float4(B + i*4);
          float4 c = add_float4(a, b);
          store_float4(C + i* 4, c);
      }
  }
  ```

  Implicit requirements: memory of A,B,C needs to be aligned to 4 floating number = 16 bytes.

  for example address 0x0012 is not aligned. Therefore we need to call align allocation instead of malloc which may only align to 8 bytes on the 64-bit system.

- Data layout (how to store a matrix in memory)

  ```python
  # Row major:
  A[i, j] => Adata[i * A.shape[1] + j]
  
  # Column major:
  A[i, j] => Adata[j * A.shape[0] + i]
  
  # Strides format:
  A[i, j] => Adata[i * A.strides[0] + j * A.strides[1]]
  ```
  Discussion about strides
    - Advantages: can perform transformation/slicing in zero copy way
    - Disadvantages: memory access becomes not continuous (`continuous`)

- Parallelization (Executes the computation on multiple threads)

  ```c++
  void vecadd(float* A, float *B, float* C) {
      // pragma omp parallel for
      for (int i = 0; i < 64; ++i) {
          float4 a = load_float4(A + i*4);
          float4 b = load_float4(B + i*4);
          float4 c = add_float4(a, b);
          store_float4(C * 4, c);
      }
  }
  ```
  - cores number = amount of things can happen simultaneously on your processor

  - clock speed of 2.6 GHz = each core can run at  2.6 GHz

  - threading - change the order in which we do specific operations(with threads waiting to happen such that another thread can be executed aka **concurrent programming**)

Case study: matrix multiplication
- Vanilla matrix multiplication
```c++
float A[n][n], B[n][n], C[n][n];
// Compute C = dot(A, B.T)
for (int i = 0; i < n; ++i) {
	for (int j = 0; j < n; ++j) {
		C[i][j] = 0;
		for (int k = 0; k < n; ++k) {
			C[i][j] += A[i][k] * B[j][k];
		}
    }     
}
```

- Memory hierarchy on modern CPUs considering latency(DRAM, L1/L2 Cache, Registers)
- Architecture aware analysis

```c++
dram float A[n][n], B[n][n], C[n][n];
// Load cost: 2 * dramspeed * n^3
// Register cost: 3
for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
        register float c = 0;
        for (int k = 0; k < n; ++k) {
            register float a = A[i][k];
            register float b = B[j][k];
            c += a * b;
        }
        C[i][j] = c;
    }
}
```

- Register tiled matrix multiplication ([example](https://lumetta.web.engr.illinois.edu/408-S20/slide-copies/ece408-lecture5-S20.pdf))