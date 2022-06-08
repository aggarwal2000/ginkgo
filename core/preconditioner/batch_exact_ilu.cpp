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

#include <ginkgo/core/preconditioner/batch_exact_ilu.hpp>


#include "core/factorization/batch_factorization_kernels.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_exact_ilu_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace batch_exact_ilu {
namespace {


GKO_REGISTER_OPERATION(check_diag_entries_exist,
                       csr::check_diagonal_entries_exist);
GKO_REGISTER_OPERATION(find_diag_locs, csr::find_diagonal_entries_locations);
GKO_REGISTER_OPERATION(generate_exact_ilu0,
                       batch_exact_ilu::generate_exact_ilu0);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l_u,
                       factorization::initialize_row_ptrs_l_u);
GKO_REGISTER_OPERATION(
    generate_common_pattern_to_fill_l_and_u,
    batch_factorization::generate_common_pattern_to_fill_l_and_u);
GKO_REGISTER_OPERATION(initialize_batch_l_and_batch_u,
                       batch_factorization::initialize_batch_l_and_batch_u);


}  // namespace
}  // namespace batch_exact_ilu


template <typename ValueType, typename IndexType>
void BatchExactIlu<ValueType, IndexType>::generate(
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
    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));

    bool all_diags{false};
    exec->run(batch_exact_ilu::make_check_diag_entries_exist(first_csr.get(),
                                                             all_diags));
    if (!all_diags) {
        // TODO: Add a diagonal addition kernel.
        throw std::runtime_error("Matrix does not have all diagonal entries!");
    }

    this->factorized_mat_ = gko::clone(this->get_executor(), sys_csr);
    this->diag_locations_ = gko::Array<index_type>(this->get_executor(), nrows);

    exec->run(batch_exact_ilu::make_find_diag_locs(first_csr.get(),
                                                   diag_locations_.get_data()));
    exec->run(batch_exact_ilu::make_generate_exact_ilu0(
        diag_locations_.get_const_data(), factorized_mat_.get()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchExactIlu<ValueType, IndexType>::transpose()
    const GKO_NOT_IMPLEMENTED;

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp>
BatchExactIlu<ValueType, IndexType>::conj_transpose() const GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
std::pair<std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>,
          std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>>
BatchExactIlu<ValueType,
              IndexType>::generate_split_factors_from_factored_matrix()
{
    const std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
        factorized_mat = this->get_const_factorized_mat();
    using unbatch_type = matrix::Csr<ValueType, IndexType>;

    auto exec = this->get_executor();
    const auto nbatch = factorized_mat->get_num_batch_entries();
    const auto nrows = factorized_mat->get_size().at(0)[0];
    const auto nnz = factorized_mat->get_num_stored_elements() / nbatch;

    // extract the first matrix, as a view, into a regular Csr matrix.
    const auto unbatch_size =
        gko::dim<2>{nrows, factorized_mat->get_size().at(0)[1]};
    auto sys_rows_view = Array<IndexType>::const_view(
        exec, nrows + 1, factorized_mat->get_const_row_ptrs());
    auto sys_cols_view = Array<IndexType>::const_view(
        exec, nnz, factorized_mat->get_const_col_idxs());
    auto sys_vals_view = Array<ValueType>::const_view(
        exec, nnz, factorized_mat->get_const_values());

    auto first_csr = unbatch_type::create_const(
        exec, unbatch_size, std::move(sys_vals_view), std::move(sys_cols_view),
        std::move(sys_rows_view));

    // find row pointers L and U and allocate memory
    Array<IndexType> l_row_ptrs(exec, nrows + 1);
    Array<IndexType> u_row_ptrs(exec, nrows + 1);

    // TODO: Write a kernel which makes use of diag info found in generate step
    // (add it to factorization kernels)
    exec->run(batch_exact_ilu::make_initialize_row_ptrs_l_u(
        first_csr.get(), l_row_ptrs.get_data(), u_row_ptrs.get_data()));

    const auto l_nnz =
        exec->copy_val_to_host(&l_row_ptrs.get_const_data()[nrows]);
    const auto u_nnz =
        exec->copy_val_to_host(&u_row_ptrs.get_const_data()[nrows]);

    std::shared_ptr<matrix_type> l_factor =
        gko::share(matrix_type::create(exec, nbatch, unbatch_size, l_nnz));
    std::shared_ptr<matrix_type> u_factor =
        gko::share(matrix_type::create(exec, nbatch, unbatch_size, u_nnz));

    exec->copy(nrows + 1, l_row_ptrs.get_const_data(),
               l_factor->get_row_ptrs());
    exec->copy(nrows + 1, u_row_ptrs.get_const_data(),
               u_factor->get_row_ptrs());

    // fill batch_L and batch_U col_idxs and values
    Array<IndexType> l_col_holders(exec, l_nnz);
    Array<IndexType> u_col_holders(exec, u_nnz);

    exec->run(batch_exact_ilu::make_generate_common_pattern_to_fill_l_and_u(
        first_csr.get(), l_factor->get_const_row_ptrs(),
        u_factor->get_const_row_ptrs(), l_col_holders.get_data(),
        u_col_holders.get_data()));

    exec->run(batch_exact_ilu::make_initialize_batch_l_and_batch_u(
        factorized_mat.get(), l_factor.get(), u_factor.get(),
        l_col_holders.get_const_data(), u_col_holders.get_const_data()));

    return std::pair<std::shared_ptr<const matrix_type>,
                     std::shared_ptr<const matrix_type>>{l_factor, u_factor};
}


#define GKO_DECLARE_BATCH_EXACT_ILU(ValueType) \
    class BatchExactIlu<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_EXACT_ILU);


}  // namespace preconditioner
}  // namespace gko
