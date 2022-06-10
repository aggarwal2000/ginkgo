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

#include "core/preconditioner/batch_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_isai {
namespace {


constexpr size_type default_block_size = 256;
constexpr int default_subwarpgrp_size = 32;

#include "common/cuda_hip/preconditioner/batch_isai_kernels.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void extract_dense_linear_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    IndexType* const dense_mat_pattern, int* const rhs_one_idxs,
    int* const sizes, int* const count_matches_per_row_for_all_csr_sys)
{
    const auto nrows = first_sys_csr->get_size()[0];
    const auto nnz_inv =
        inv->get_num_stored_elements() / inv->get_num_batch_entries();
    const auto nnz_given_mat = first_sys_csr->get_num_stored_elements();
    const size_type num_blocks =
        ceildiv(default_subwarpgrp_size * nnz_inv, default_block_size);

    extract_dense_sys_pattern_kernel<default_subwarpgrp_size>
        <<<num_blocks, default_block_size>>>(
            static_cast<int>(nrows), first_sys_csr->get_const_row_ptrs(),
            first_sys_csr->get_const_col_idxs(),
            static_cast<int>(nnz_given_mat), inv->get_const_row_ptrs(),
            inv->get_const_col_idxs(), static_cast<int>(nnz_inv),
            dense_mat_pattern, rhs_one_idxs, sizes,
            count_matches_per_row_for_all_csr_sys);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_values_dense_mat_and_solve(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    const IndexType* const dense_mat_pattern, const int* const rhs_one_idxs,
    const int* const sizes,
    const gko::preconditioner::batch_isai_sys_mat_type&
        given_system_matrix_type)
{
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nnz_inv = inv->get_num_stored_elements() / nbatch;
    const auto nnz_given_mat = sys_csr->get_num_stored_elements() / nbatch;

    int matrix_type;

    if (given_system_matrix_type ==
        gko::preconditioner::batch_isai_sys_mat_type::lower_tri) {
        //  std::cout << "\n small uppertrsv \n" << std::endl;
        matrix_type = 0;
    } else if (given_system_matrix_type ==
               gko::preconditioner::batch_isai_sys_mat_type::upper_tri) {
        // std::cout << "\n small lower trsv \n" << std::endl;
        matrix_type = 1;
    } else if (given_system_matrix_type ==
               gko::preconditioner::batch_isai_sys_mat_type::general) {
        //  std::cout << "\n small general isai \n" << std::endl;
        matrix_type = 2;
    } else {
        GKO_NOT_IMPLEMENTED;
    }

    const size_type num_blocks =
        ceildiv(default_subwarpgrp_size * nrows * nbatch, default_block_size);

    fill_values_dense_mat_and_solve_kernel<default_subwarpgrp_size>
        <<<num_blocks, default_block_size>>>(
            nbatch, static_cast<int>(nrows), inv->get_const_row_ptrs(),
            inv->get_const_col_idxs(), as_cuda_type(inv->get_values()),
            static_cast<int>(nnz_inv), sys_csr->get_const_row_ptrs(),
            sys_csr->get_const_col_idxs(),
            as_cuda_type(sys_csr->get_const_values()),
            static_cast<int>(nnz_given_mat), dense_mat_pattern, rhs_one_idxs,
            sizes, matrix_type);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL);


// only used for testing purpose
template <typename ValueType, typename IndexType>
void apply_isai(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::BatchCsr<ValueType, IndexType>* const inv_mat,
                const matrix::BatchDense<ValueType>* const r,
                matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL);


}  // namespace batch_isai
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
