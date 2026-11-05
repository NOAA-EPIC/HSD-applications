# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CCACHE_DIR=/root/.ccache
ENV OMP_NUM_THREADS=4
ENV OMP_PROC_BIND=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    ccache \
    libomp-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy prototype source into container
COPY . .

ARG ENABLE_CUDA=OFF
ARG CMAKE_BUILD_TYPE=Release

# Configure with ccache, Serial, and OpenMP (plus CUDA if enabled)
RUN cmake -B build -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_CUDA=${ENABLE_CUDA}

# Build prototype executable
RUN cmake --build build --parallel

# Execute CTest test suite at build time (runs CPU tests by default)
RUN ctest --test-dir build --verbose --output-on-failure -L cpu

CMD ["/workspace/build/kokkos_runner"]

