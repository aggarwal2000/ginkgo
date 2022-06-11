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
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>
#include <ginkgo/core/solver/batch_bicgstab.hpp>

#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/preconditioner/batch_isai_kernels.hpp"
#include "core/preconditioner/isai_kernels.hpp"

namespace gko {
namespace preconditioner {
namespace batch_isai {
namespace {


GKO_REGISTER_OPERATION(extract_dense_linear_sys_pattern,
                       batch_isai::extract_dense_linear_sys_pattern);
GKO_REGISTER_OPERATION(fill_values_dense_mat_and_solve,
                       batch_isai::fill_values_dense_mat_and_solve);


GKO_REGISTER_OPERATION(find_cumulative_nnz_csr_matrices,
                       batch_isai::find_cumulative_nnz_csr_matrices);
GKO_REGISTER_OPERATION(
    set_row_ptrs_csr_matrix_and_extract_csr_pattern,
    batch_isai::set_row_ptrs_csr_matrix_and_extract_csr_pattern);
GKO_REGISTER_OPERATION(csr_fill_values_using_pattern,
                       batch_isai::csr_fill_values_using_pattern);
GKO_REGISTER_OPERATION(initialize_batched_rhs_and_sol,
                       batch_isai::initialize_batched_rhs_and_sol);
GKO_REGISTER_OPERATION(write_large_sys_solution_to_inverse,
                       batch_isai::write_large_sys_solution_to_inverse);


template <typename ValueType, typename IndexType>
void allocate_memory_and_set_row_ptrs_and_col_idxs_according_to_the_chosen_spy_pattern(
    std::shared_ptr<const Executor> exec,
    std::shared_ptr<const matrix::Csr<ValueType, IndexType>> first_sys_csr,
    const int spy_power, const size_type nbatch,
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>>& left_isai)
{
    using csr = matrix::Csr<ValueType, IndexType>;
    using batch_csr = matrix::BatchCsr<ValueType, IndexType>;

    std::shared_ptr<csr> tmp =
        gko::preconditioner::extend_sparsity(exec, first_sys_csr, spy_power);

    const auto nrows = tmp->get_size()[0];
    const auto nnz = tmp->get_num_stored_elements();
    left_isai =
        gko::share(batch_csr::create(exec, nbatch, tmp->get_size(), nnz));
    exec->copy(nrows + 1, tmp->get_const_row_ptrs(), left_isai->get_row_ptrs());
    exec->copy(nnz, tmp->get_const_col_idxs(), left_isai->get_col_idxs());
}

template <typename ValueType, typename IndexType>
void extract_and_solve_csr_linear_sys(
    std::shared_ptr<const Executor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* given_sys_csr_A,
    const Array<int>& sizes, const Array<int>& rhs_one_idxs,
    gko::preconditioner::batch_isai_sys_mat_type sys_matrix_type,
    Array<int>& count_matches_per_row_for_all_csr_sys,
    Array<int>& large_csr_linear_sys_nnz,
    matrix::BatchCsr<ValueType, IndexType>* inv_A)
{
    using batch_csr = matrix::BatchCsr<ValueType, IndexType>;
    using batch_dense = matrix::BatchDense<ValueType>;
    using csr = matrix::Csr<ValueType, IndexType>;

    exec->run(batch_isai::make_find_cumulative_nnz_csr_matrices(
        inv_A, sizes.get_const_data(),
        count_matches_per_row_for_all_csr_sys.get_data(),
        large_csr_linear_sys_nnz.get_data()));


    // TODO: Avoid extra copy by using views if exec is Refernce/Omp executor
    // (i.e. use a copy only when exec's allocation space is different than that
    // of exec->master())
    const Array<int> large_csr_linear_sys_nnz_host(exec->get_master(),
                                                   large_csr_linear_sys_nnz);
    const Array<int> sizes_host(exec->get_master(), sizes);
    const Array<int> rhs_one_idxs_host(exec->get_master(), rhs_one_idxs);

    const auto nrows = given_sys_csr_A->get_size().at(0)[0];
    const auto nbatch = given_sys_csr_A->get_num_batch_entries();

    for (int i_row_idx = 0; i_row_idx < static_cast<int>(nrows); i_row_idx++) {
        const int i_size = sizes_host.get_const_data()[i_row_idx];

        if (i_size <= gko::kernels::batch_isai::row_size_limit) {
            assert(large_csr_linear_sys_nnz_host.get_const_data()[i_row_idx] ==
                   -1);
            continue;
        }

        // std::cout << "Linear sys index: " << i_row_idx << " Extract and solve
        // csr sys " << std::endl;
        // Extract CSR pattern, transpose, fill vals and solve batched linear
        // sys, write solution back to batched_inv

        const int nnz_csr_sys =
            large_csr_linear_sys_nnz_host.get_const_data()[i_row_idx];
        std::shared_ptr<batch_csr> large_csr_system = share(batch_csr::create(
            exec, nbatch, dim<2>(i_size, i_size), nnz_csr_sys));

        Array<IndexType> values_pattern_large_csr_system(exec, nnz_csr_sys);

        exec->run(
            batch_isai::make_set_row_ptrs_csr_matrix_and_extract_csr_pattern(
                i_size, i_row_idx, given_sys_csr_A, inv_A,
                count_matches_per_row_for_all_csr_sys.get_const_data(),
                large_csr_system.get(),
                values_pattern_large_csr_system.get_data()));

        Array<IndexType> values_pattern_large_csr_system_view =
            Array<IndexType>::view(exec, nnz_csr_sys,
                                   values_pattern_large_csr_system.get_data());
        Array<IndexType> row_ptrs_large_csr_system_view =
            Array<IndexType>::view(exec, i_size + 1,
                                   large_csr_system->get_row_ptrs());
        Array<IndexType> col_idxs_large_csr_system_view =
            Array<IndexType>::view(exec, nnz_csr_sys,
                                   large_csr_system->get_col_idxs());

        // NOTE: IndexType (in place of ValueType) causing problems
        // No explicit instantiation for that, so such CSR matrix class is not
        // available
        /*
        // create a csr matrix which does not deallocate the 3 arrays (row_ptrs,
        // col_idxs and vals)
        std::shared_ptr<matrix::Csr<IndexType, IndexType>> csr_pattern_matrix =
            gko::share(matrix::Csr<IndexType, IndexType>::create(
                exec, dim<2>(i_size, i_size),
                std::move(values_pattern_large_csr_system_view),
                std::move(col_idxs_large_csr_system_view),
                std::move(row_ptrs_large_csr_system_view)));
        // null/view deletor of these arrays is also copied, right? - look at
        // the Csr matrix and Array constructors(otheriwse, there would be a
        // double free error)

        csr_pattern_matrix->transpose();
        */

        exec->run(batch_isai::make_csr_fill_values_using_pattern(
            values_pattern_large_csr_system.get_const_data(),
            large_csr_system.get(), given_sys_csr_A));

        // create and initialize batched rhs and sol x
        std::shared_ptr<batch_dense> rhs_batch = gko::share(
            batch_dense::create(exec, batch_dim<2>(nbatch, dim<2>(i_size, 1))));
        assert(rhs_one_idxs_host.get_const_data()[i_row_idx] >= 0);
        assert(i_size > rhs_one_idxs_host.get_const_data()[i_row_idx]);
        std::shared_ptr<batch_dense> x_batch = gko::share(
            batch_dense::create(exec, batch_dim<2>(nbatch, dim<2>(i_size, 1))));
        exec->run(batch_isai::make_initialize_batched_rhs_and_sol(
            rhs_one_idxs_host.get_const_data()[i_row_idx], rhs_batch.get(),
            x_batch.get()));

        if (sys_matrix_type ==
            gko::preconditioner::batch_isai_sys_mat_type::lower_tri) {
            // call batched upper trsv (TODO: Implement upper triangular batched
            // solver)
            GKO_NOT_IMPLEMENTED;
        } else if (sys_matrix_type ==
                   gko::preconditioner::batch_isai_sys_mat_type::upper_tri) {
            // call batched lower trsv (TODO: Implement lower triangular batched
            // solver)
            GKO_NOT_IMPLEMENTED;
        } else if (sys_matrix_type ==
                   gko::preconditioner::batch_isai_sys_mat_type::general) {
            // call batched bicgstab solver with scalar jacobi preconditioner
            auto jacobi_preconditioned_batch_bicgstab_solver =
                gko::solver::BatchBicgstab<ValueType>::build()
                    .with_preconditioner(
                        gko::preconditioner::BatchJacobi<ValueType,
                                                         IndexType>::build()
                            .on(exec))
                    .on(exec)
                    ->generate(large_csr_system);

            jacobi_preconditioned_batch_bicgstab_solver->apply(rhs_batch.get(),
                                                               x_batch.get());
        } else {
            GKO_NOT_IMPLEMENTED;
        }

        exec->run(batch_isai::make_write_large_sys_solution_to_inverse(
            i_size, i_row_idx, inv_A, x_batch.get()));
    }
}


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
    dense_mat_pattern.fill(-1);
    gko::Array<int> rhs_one_idxs(exec, nrows);
    rhs_one_idxs.fill(-1);
    gko::Array<int> sizes(exec, nrows);
    gko::Array<int> large_csr_linear_sys_nnz(exec, nrows);
    gko::Array<int> count_matches_per_row_for_all_csr_sys(
        exec, this->left_isai_->get_num_stored_elements() / nbatch);

    exec->run(batch_isai::make_extract_dense_linear_sys_pattern(
        first_csr.get(), this->left_isai_.get(), dense_mat_pattern.get_data(),
        rhs_one_idxs.get_data(), sizes.get_data(),
        count_matches_per_row_for_all_csr_sys.get_data()));

    exec->run(batch_isai::make_fill_values_dense_mat_and_solve(
        sys_csr, this->left_isai_.get(), dense_mat_pattern.get_const_data(),
        rhs_one_idxs.get_const_data(), sizes.get_const_data(),
        this->parameters_.matrix_type_isai));

    // batch_isai::extract_and_solve_csr_linear_sys(
    //     exec, sys_csr, sizes, rhs_one_idxs,
    //     this->parameters_.matrix_type_isai,
    //     count_matches_per_row_for_all_csr_sys, large_csr_linear_sys_nnz,
    //     this->left_isai_.get());
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


#define GKO_DECLARE_BATCH_ISAI(ValueType) class BatchIsai<ValueType, int32>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_ISAI);

}  // namespace preconditioner
}  // namespace gko
