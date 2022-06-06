/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/preconditioner/batch_par_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>

#include "core/matrix/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_par_ilu {
namespace {


constexpr size_type default_block_size = 256;

#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_par_ilu_kernels.hpp.inc"
//#include "common/cuda_hip/preconditioner/batch_par_ilu.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void compute_par_ilu0(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const int num_sweeps, const IndexType* const dependencies,
    const IndexType* const nz_ptrs)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto nnz =
        static_cast<int>(sys_mat->get_num_stored_elements() / nbatch);
    const auto l_nnz =
        static_cast<int>(l_factor->get_num_stored_elements() / nbatch);
    const auto u_nnz =
        static_cast<int>(u_factor->get_num_stored_elements() / nbatch);

    const size_type dynamic_shared_mem_bytes =
        (l_nnz + u_nnz) * sizeof(ValueType);

    compute_parilu0_kernel<<<nbatch, default_block_size,
                             dynamic_shared_mem_bytes>>>(
        nbatch, num_rows, nnz, as_cuda_type(sys_mat->get_const_values()), l_nnz,
        as_cuda_type(l_factor->get_values()), u_nnz,
        as_cuda_type(u_factor->get_values()), num_sweeps, dependencies,
        nz_ptrs);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_COMPUTE_PARILU0_KERNEL);


template <typename ValueType, typename IndexType>
void apply_par_ilu0(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    const matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_APPLY_KERNEL);


}  // namespace batch_par_ilu
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
