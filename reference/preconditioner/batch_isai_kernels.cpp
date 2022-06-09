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


#include <algorithm>
#include <cassert>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
//#include "reference/preconditioner/batch_isai.hpp"
#include "reference/preconditioner/batch_trsv.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace batch_isai {


//#include "reference/preconditioner/batch_exact_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void extract_dense_linear_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    IndexType* const dense_mat_pattern, int* const rhs_one_idxs,
    int* const sizes,
    int* const count_matches_per_row_for_all_csr_sys) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_values_dense_mat_and_solve(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    const IndexType* const dense_mat_pattern, const int* const rhs_one_idxs,
    const int* const sizes,
    const gko::preconditioner::batch_isai_sys_mat_type& matrix_type_isai)
    GKO_NOT_IMPLEMENTED;

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
}  // namespace reference
}  // namespace kernels
}  // namespace gko
