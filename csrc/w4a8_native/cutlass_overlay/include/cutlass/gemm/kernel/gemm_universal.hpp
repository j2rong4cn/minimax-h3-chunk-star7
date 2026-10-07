#pragma once
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include <cute/util/type_traits.hpp>
#include <cute/tensor_impl.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
namespace cutlass::gemm::kernel {
template <class ProblemShape, class = void>
struct IsCutlass3ArrayKernel : cute::false_type {};
template <typename ProblemShape>
struct IsCutlass3ArrayKernel<ProblemShape, cute::void_t<typename ProblemShape::UnderlyingProblemShape>> : cute::true_type {};
}
