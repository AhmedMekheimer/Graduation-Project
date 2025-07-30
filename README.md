# GPU-Accelerated AUTOSAR Transformers

This project enhances the performance of AUTOSAR COM-based transformers by leveraging GPU acceleration with CUDA C programming. The work was sponsored by Siemens and graded "A-".

## Project Overview

The project focuses on optimizing serialization and deserialization operations in AUTOSAR applications, where:
- **Serialization**: Transforms a struct of different data types (complex data) to an array of bytes (buffer array)
- **Deserialization**: Performs the reverse transformation

Key achievements include:
1. Implemented GPU-accelerated versions of COM-based transformer APIs
2. Optimized memory allocation strategies comparing CPU and GPU approaches
3. Developed performance benchmarks comparing different implementations

## Technical Approach

### AUTOSAR Architecture Understanding
- Studied AUTOSAR layered architecture and general transformers through AUTOSAR documentation
- Analyzed COM-based transformers' configurations, type definitions, and APIs

### GPU Acceleration Implementation
- Leveraged CUDA C programming to utilize GPU parallel processing capabilities
- Implemented thread hierarchy optimization (blocks and grids) for maximum throughput
- Evaluated different memory allocation strategies:
  - CPU dynamic memory allocation
  - CUDA unified memory allocation (using both GPU and CPU memory)

## Performance Results

The project includes comprehensive performance analysis showing:
- Comparison graphs between CPU and GPU implementations
- Memory allocation performance benchmarks
- Optimal data distribution strategies for GPU threads
- Limitations of GPU thread utilization for this use case

Key findings demonstrate significant performance improvements in the GPU-accelerated versions compared to traditional CPU implementations.



