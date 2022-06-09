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

#include <ginkgo/core/preconditioner/batch_isai.hpp>

#include "core/matrix/batch_csr_kernels.hpp"
#include "core/preconditioner/batch_isai_kernels.hpp"
#include "core/preconditioner/isai_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace batch_isai {
namespace {

template <typename ValueType, typename IndexType>
void allocate_memory_and_set_row_ptrs_and_col_idxs_according_to_the_chosen_spy_pattern(
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> first_sys_csr,
    const int spy_power, const size_type nbatch,
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>>& left_isai)
{
    using csr = matrix::Csr<ValueType, IndexType>;
    using batch_csr = matrix::BatchCsr<ValueType, IndexType>;
    auto exec = first_sys_csr->get_executor();

    std::shared_ptr<csr> tmp =
        gko::preconditioner::extend_sparsity(exec, first_sys_csr, spy_power);

    const auto nrows = tmp->get_size()[0];
    const auto nnz = tmp->get_num_stored_elements();
    left_isai =
        gko::share(batch_csr::create(exec, nbatch, tmp->get_size(), nnz));
    exec->copy(nrows + 1, tmp->get_const_row_ptrs(), left_isai->get_row_ptrs());
    exec->copy(nnz, tmp->get_const_col_idxs(), left_isai->get_col_idxs());
}

GKO_REGISTER_OPERATION(extract_dense_linear_sys_pattern,
                       batch_isai::extract_dense_linear_sys_pattern);
GKO_REGISTER_OPERATION(fill_values_dense_mat_and_solve,
                       batch_isai::fill_values_dense_mat_and_solve);


}  // namespace
}  // namespace batch_isai


template <typename ValueType, typename IndexType>
void BatchIsai<ValueType, IndexType>::generate(
    const BatchLinOp* const system_matrix)
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    // generate entire batch of factorizations
    if (!system_matrix->get_size().stores_equal_sizes()) {
        GKO_NOT_IMPLEMENTED;
    }
    auto exec = this->get_executor();
    const matrix_type* sys_csr{};
    auto a_matrix = matrix_type::create(exec);
    if (auto temp_csr = dynamic_cast<const matrix_type*>(system_matrix)) {
        sys_csr = temp_csr;
    } else {
        as<ConvertibleTo<matrix_type>>(system_matrix)
            ->convert_to(a_matrix.get());
        sys_csr = a_matrix.get();
    }

    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;

    std::shared_ptr<matrix_type> temp_sys_csr_smart;
    if (parameters_.skip_sorting == false) {
        temp_sys_csr_smart = gko::clone(this->get_executor(), sys_csr);
        temp_sys_csr_smart->sort_by_column_index();
        sys_csr = temp_sys_csr_smart.get();
    }

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size = gko::dim<2>{nrows, sys_csr->get_size().at(0)[1]};
    auto sys_rows_view = Array<IndexType>::const_view(
        exec, nrows + 1, sys_csr->get_const_row_ptrs());
    auto sys_cols_view =
        Array<IndexType>::const_view(exec, nnz, sys_csr->get_const_col_idxs());
    auto sys_vals_view =
        Array<ValueType>::const_view(exec, nnz, sys_csr->get_const_values());
    std::shared_ptr<const unbatch_type> first_csr =
        gko::share(unbatch_type::create_const(
            exec, unbatch_size, std::move(sys_vals_view),
            std::move(sys_cols_view), std::move(sys_rows_view)));


    batch_isai::
        allocate_memory_and_set_row_ptrs_and_col_idxs_according_to_the_chosen_spy_pattern(
            exec, first_csr, this->parameters_.sparsity_power, nbatch,
            this->left_isai_);

    gko::Array<IndexType> dense_mat_pattern(
        exec, gko::kernels::batch_isai::row_size_limit *
                  gko::kernels::batch_isai::row_size_limit * nrows);
    gko::Array<int> rhs_one_idxs(exec, nrows);
    gko::Array<int> sizes(exec, nrows);
    gko::Array<int> large_csr_linear_sys_nnz(exec, nrows);
    gko::Array<int> count_matches_per_row_for_all_csr_sys(
        exec, this->left_isai_->get_num_stored_elements() / nbatch);

    exec->run(batch_isai::make_extract_dense_linear_sys_pattern(
        sys_csr, this->left_isai_.get(), dense_mat_pattern.get_data(),
        rhs_one_idxs.get_data(), sizes.get_data(),
        count_matches_per_row_for_all_csr_sys.get_data()));

    exec->run(batch_isai::make_fill_values_dense_mat_and_solve(
        sys_csr, this->left_isai_.get(), dense_mat_pattern.get_const_data(),
        rhs_one_idxs.get_const_data(), sizes.get_const_data(),
        this->parameters_.matrix_type_isai));

    // TODO: Add kernel: extract and solve csr linear sys (will write a core
    // function which calls make_kernels etc...)
}

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchIsai<ValueType, IndexType>::transpose() const
{
    GKO_NOT_IMPLEMENTED;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchIsai<ValueType, IndexType>::conj_transpose()
    const
{
    GKO_NOT_IMPLEMENTED;
}


}  // namespace preconditioner
}  // namespace gko
