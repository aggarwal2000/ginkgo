/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/preconditioner/scalarjacobi_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cublas_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/pointer_mode_guard.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Scalarjacobi preconditioner namespace.
 * @ref Scalarjacobi
 * @ingroup scalarjacobi
 */
namespace scalarjacobi {


constexpr auto default_block_size = 512;

#include "common/preconditioner/scalarjacobi_kernels.hpp.inc"


template <typename ValueType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, int32> *system_matrix,
              Array<ValueType> &inv_eles)
{
    const auto num_rows = system_matrix->get_size()[0];
    const auto row_ptrs =
        system_matrix
            ->get_const_row_ptrs();  // Really need const here ? const int a =
                                     // 3; auto b = a; b is not const right, but
                                     // with pointers it is..?
    const auto col_idxs = system_matrix->get_const_col_idxs();
    const auto vals = system_matrix->get_const_values();

    auto inv = inv_eles.get_data();

    const dim3 block(default_block_size, 1, 1);

    const size_type grid_size_x =
        ceildiv(num_rows * config::warp_size, default_block_size);

    const dim3 grid(grid_size_x, 1, 1);

    kernel::generate_kernel<<<grid, block>>>(
        num_rows, as_cuda_type(row_ptrs), as_cuda_type(col_idxs),
        as_cuda_type(vals), as_cuda_type(inv));
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCALARJACOBI_GENERATE_KERNEL);


template <typename ValueType>
void simple_apply(std::shared_ptr<const CudaExecutor> exec,
                  const Array<ValueType> &inv_eles,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x)
{
    const auto num_rows = b->get_size()[0];
    const auto num_cols = b->get_size()[1];
    const auto stride_source = b->get_stride();
    const auto stride_result = x->get_stride();

    const auto inv = inv_eles.get_const_data();
    auto result = x->get_values();
    const auto source = b->get_const_values();

    const dim3 block(default_block_size, 1, 1);

    const dim3 grid(ceildiv(num_rows * num_cols, default_block_size), 1, 1);

    kernel::simple_apply_kernel<<<grid, block>>>(
        num_rows, num_cols, stride_source, as_cuda_type(source), stride_result,
        as_cuda_type(result), as_cuda_type(inv));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_SCALARJACOBI_SIMPLE_APPLY_KERNEL);


}  // namespace scalarjacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
