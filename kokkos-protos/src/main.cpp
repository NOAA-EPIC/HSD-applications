#include <Kokkos_Core.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <cstdlib>

#if defined(KOKKOS_ENABLE_OPENMP)
#include <omp.h>
#endif

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

// Generic vector add and verification routine for any Kokkos execution space
template <typename ExecSpace>
bool run_backend_test(const std::string& name, int N = 100000) {
    std::cout << "\n----------------------------------------\n";
    std::cout << "Testing Backend: " << name << "\n";
    std::cout << "Execution Space: " << ExecSpace().name() << "\n";
    std::cout << "Memory Space:    " << typename ExecSpace::memory_space().name() << "\n";
    std::cout << "Vector Size:     " << N << " elements\n";

#if defined(KOKKOS_ENABLE_OPENMP)
    if (std::string(ExecSpace().name()) == "OpenMP") {
        std::cout << "OpenMP Max Threads: " << omp_get_max_threads() << "\n";
    }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
    if (std::string(ExecSpace().name()) == "Cuda") {
        int device_id = 0;
        cudaError_t err = cudaGetDevice(&device_id);
        if (err == cudaSuccess) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device_id);
            std::cout << "CUDA Device [" << device_id << "]: " << prop.name
                      << " (Compute Capability " << prop.major << "." << prop.minor << ")\n";
        } else {
            std::cerr << "Warning: Could not query CUDA device properties.\n";
        }
    }
#endif

    using MemSpace = typename ExecSpace::memory_space;
    Kokkos::View<double*, MemSpace> A("A", N);
    Kokkos::View<double*, MemSpace> B("B", N);
    Kokkos::View<double*, MemSpace> C("C", N);

    // Initialize inputs: A[i] = 1.25, B[i] = 2.75
    Kokkos::parallel_for("init_A", Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(int i) {
        A(i) = 1.25;
    });
    Kokkos::parallel_for("init_B", Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(int i) {
        B(i) = 2.75;
    });
    ExecSpace().fence();

    // Vector addition: C[i] = A[i] + B[i]
    Kokkos::parallel_for("vec_add", Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(int i) {
        C(i) = A(i) + B(i);
    });
    ExecSpace().fence();

    // Verify: expected C[i] == 4.0
    int errors = 0;
    Kokkos::parallel_reduce("verify", Kokkos::RangePolicy<ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i, int& lerr) {
            if (std::abs(C(i) - 4.0) > 1e-12) {
                lerr++;
            }
        }, errors);
    ExecSpace().fence();

    if (errors == 0) {
        std::cout << "Status: PASS (0 errors across " << N << " elements)\n";
        return true;
    } else {
        std::cerr << "Status: FAIL (" << errors << " mismatched elements)\n";
        return false;
    }
}

int main(int argc, char* argv[]) {
    std::string target_backend = "all";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--backend=", 0) == 0) {
            target_backend = arg.substr(10);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: kokkos_runner [--backend=serial|openmp|cuda|all]\n";
            return 0;
        }
    }

    Kokkos::ScopeGuard guard(argc, argv);

    bool all_passed = true;
    bool ran_any = false;

    // Serial
    if (target_backend == "all" || target_backend == "serial") {
#if defined(KOKKOS_ENABLE_SERIAL)
        ran_any = true;
        if (!run_backend_test<Kokkos::Serial>("Serial")) {
            all_passed = false;
        }
#else
        std::cerr << "Error: Kokkos Serial backend was not enabled at compile time.\n";
        return 1;
#endif
    }

    // OpenMP
    if (target_backend == "all" || target_backend == "openmp") {
#if defined(KOKKOS_ENABLE_OPENMP)
        ran_any = true;
        if (!run_backend_test<Kokkos::OpenMP>("OpenMP")) {
            all_passed = false;
        }
#else
        if (target_backend == "openmp") {
            std::cerr << "Error: Kokkos OpenMP backend was not enabled at compile time.\n";
            return 1;
        }
#endif
    }

    // CUDA
    if (target_backend == "all" || target_backend == "cuda") {
#if defined(KOKKOS_ENABLE_CUDA)
        ran_any = true;
        if (!run_backend_test<Kokkos::Cuda>("CUDA")) {
            all_passed = false;
        }
#else
        if (target_backend == "cuda") {
            std::cerr << "Error: Kokkos CUDA backend was not enabled at compile time.\n";
            return 1;
        }
#endif
    }

    if (!ran_any) {
        std::cerr << "Unknown backend requested: " << target_backend << "\n";
        return 1;
    }

    std::cout << "\nOverall Test Result: " << (all_passed ? "SUCCESS" : "FAILED") << "\n";
    return all_passed ? 0 : 1;
}

