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

#include <typeinfo>
#include "core/matrix/cvcsr_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
/**
 * @brief The CUDA namespace.
 *
 * @ingroup cuda
 */
namespace cuda {
/**
 * @brief The Cvcsrrdinate matrix format namespace.
 *
 * @ingroup cvcsr
 */
namespace cvcsr {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;
constexpr int wsize = config::warp_size;
constexpr int classical_overweight = 32;


using classical_kernels =
    syn::value_list<int, config::warp_size, 32, 16, 8, 4, 2, 1>;


#include "common/matrix/cvcsr_kernels.hpp.inc"


namespace host_kernel {


template <int dim, typename Type1, typename Type2>
GKO_INLINE auto as_cuda_accessor(
    const range<accessor::reduced_row_major<dim, Type1, Type2>> &acc)
{
    return range<
        accessor::reduced_row_major<dim, cuda_type<Type1>, cuda_type<Type2>>>(
        acc.get_accessor().get_size(),
        as_cuda_type(acc.get_accessor().get_stored_data()),
        acc.get_accessor().get_stride());
}


template <int subwarp_size, typename ValueType, typename StorageType,
          typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Cvcsr<ValueType, StorageType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c,
                    const matrix::Dense<ValueType> *alpha = nullptr,
                    const matrix::Dense<ValueType> *beta = nullptr)
{
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * classical_overweight;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const dim3 block(spmv_block_size);

    if (alpha == nullptr && beta == nullptr) {
        kernel::abstract_classical_spmv<subwarp_size><<<grid, block, 0, 0>>>(
            a->get_size()[0], as_cuda_accessor(a->get_const_values()),
            a->get_const_col_idxs(), as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(c->get_values()), c->get_stride());

    } else if (alpha != nullptr && beta != nullptr) {
        kernel::abstract_classical_spmv<subwarp_size><<<grid, block, 0, 0>>>(
            a->get_size()[0], as_cuda_type(alpha->get_const_values()),
            as_cuda_accessor(a->get_const_values()), a->get_const_col_idxs(),
            as_cuda_type(a->get_const_row_ptrs()),
            as_cuda_type(b->get_const_values()), b->get_stride(),
            as_cuda_type(beta->get_const_values()),
            as_cuda_type(c->get_values()), c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);

}  // namespace host_kernel


template <typename ValueType, typename StorageType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Cvcsr<ValueType, StorageType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto strategy = std::dynamic_pointer_cast<
        const typename matrix::Csr<StorageType, IndexType>::classical>(
        a->csr_->get_strategy());
    auto max_length_per_row = strategy->get_max_length_per_row();

    host_kernel::select_classical_spmv(
        classical_kernels(),
        [&max_length_per_row](int compiled_info) {
            return max_length_per_row >= compiled_info;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_STORAGE_AND_INDEX_TYPE(
    GKO_DECLARE_CVCSR_SPMV_KERNEL);


template <typename ValueType, typename StorageType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Cvcsr<ValueType, StorageType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    auto strategy = std::dynamic_pointer_cast<
        const typename matrix::Csr<StorageType, IndexType>::classical>(
        a->csr_->get_strategy());
    auto max_length_per_row = strategy->get_max_length_per_row();

    host_kernel::select_classical_spmv(
        classical_kernels(),
        [&max_length_per_row](int compiled_info) {
            return max_length_per_row >= compiled_info;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, aplha, beta);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_STORAGE_AND_INDEX_TYPE(
    GKO_DECLARE_CVCSR_ADVANCED_SPMV_KERNEL);


}  // namespace cvcsr
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
