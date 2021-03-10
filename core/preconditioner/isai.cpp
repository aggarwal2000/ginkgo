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

#include <ginkgo/core/preconditioner/isai.hpp>


#include <functional>
#include <memory>
#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/preconditioner/jacobi.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/solver/lower_trs.hpp>
#include <ginkgo/core/solver/upper_trs.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/base/utils.hpp"
#include "core/factorization/factorization_kernels.hpp"
#include "core/preconditioner/isai_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace isai {


GKO_REGISTER_OPERATION(generate_tri_inverse, isai::generate_tri_inverse);
GKO_REGISTER_OPERATION(generate_general_inverse,
                       isai::generate_general_inverse);
GKO_REGISTER_OPERATION(generate_excess_system, isai::generate_excess_system);
GKO_REGISTER_OPERATION(scale_excess_solution, isai::scale_excess_solution);
GKO_REGISTER_OPERATION(scatter_excess_solution, isai::scatter_excess_solution);
GKO_REGISTER_OPERATION(initialize_row_ptrs_l,
                       factorization::initialize_row_ptrs_l);
GKO_REGISTER_OPERATION(initialize_l, factorization::initialize_l);


}  // namespace isai


/**
 * @internal
 *
 * Helper function that extends the sparsity pattern of the matrix M to M^n
 * without changing its values.
 *
 * The input matrix must be sorted and on the correct executor for this to work.
 * If `power` is 1, the matrix will be returned unchanged.
 */
template <typename Csr>
std::shared_ptr<Csr> extend_sparsity(std::shared_ptr<const Executor> &exec,
                                     std::shared_ptr<const Csr> mtx, int power)
{
    GKO_ASSERT_EQ(power >= 1, true);
    if (power == 1) {
        // copy the matrix, as it will be used to store the inverse
        return {std::move(mtx->clone())};
    }
    auto id_power = mtx->clone();
    auto tmp = Csr::create(exec, mtx->get_size());
    // accumulates mtx * the remainder from odd powers
    auto acc = mtx->clone();
    // compute id^(n-1) using square-and-multiply
    int i = power - 1;
    while (i > 1) {
        if (i % 2 != 0) {
            // store one power in acc:
            // i^(2n+1) -> i*i^2n
            id_power->apply(lend(acc), lend(tmp));
            std::swap(acc, tmp);
            i--;
        }
        // square id_power: i^2n -> (i^2)^n
        id_power->apply(lend(id_power), lend(tmp));
        std::swap(id_power, tmp);
        i /= 2;
    }
    // combine acc and id_power again
    id_power->apply(lend(acc), lend(tmp));
    return {std::move(tmp)};
}


template <isai_type IsaiType, typename ValueType, typename IndexType,
          typename StorageType>
void Isai<IsaiType, ValueType, IndexType, StorageType>::generate_inverse(
    std::shared_ptr<const LinOp> input, bool skip_sorting, int power,
    IndexType excess_limit)
{
    using Dense = matrix::Dense<ValueType>;
    using LowerTrs = solver::LowerTrs<ValueType, IndexType>;
    using UpperTrs = solver::UpperTrs<ValueType, IndexType>;
    using Gmres = solver::Gmres<ValueType>;
    using bj = preconditioner::Jacobi<ValueType, IndexType>;
    GKO_ASSERT_IS_SQUARE_MATRIX(input);
    auto exec = this->get_executor();
    auto is_lower = IsaiType == isai_type::lower;
    auto is_general = IsaiType == isai_type::general;
    auto is_spd = IsaiType == isai_type::spd;
    auto to_invert = convert_to_with_sorting<Csr>(exec, input, skip_sorting);
    auto num_rows = to_invert->get_size()[0];
    std::shared_ptr<Csr> inverted;
    if (!is_spd) {
        inverted = extend_sparsity(exec, to_invert, power);
    } else {
        // Extract lower triangular part: compute non-zeros
        Array<IndexType> inverted_row_ptrs{exec, num_rows + 1};
        exec->run(isai::make_initialize_row_ptrs_l(
            to_invert.get(), inverted_row_ptrs.get_data()));

        // Get nnz from device memory
        auto inverted_nnz = static_cast<size_type>(
            exec->copy_val_to_host(inverted_row_ptrs.get_data() + num_rows));

        // Init arrays
        Array<IndexType> inverted_col_idxs{exec, inverted_nnz};
        Array<ValueType> inverted_vals{exec, inverted_nnz};
        std::shared_ptr<Csr> inverted_base = Csr::create(
            exec, dim<2>{num_rows, num_rows}, std::move(inverted_vals),
            std::move(inverted_col_idxs), std::move(inverted_row_ptrs));

        // Extract lower factor: columns and values
        exec->run(isai::make_initialize_l(to_invert.get(), inverted_base.get(),
                                          false));

        inverted = power == 1
                       ? std::move(inverted_base)
                       : extend_sparsity<Csr>(exec, inverted_base, power);
    }

    // This stores the beginning of the RHS for the sparse block associated with
    // each row of inverted_l
    Array<IndexType> excess_block_ptrs{exec, num_rows + 1};
    // This stores the beginning of the non-zeros belonging to each row in the
    // system of excess blocks
    Array<IndexType> excess_row_ptrs_full{exec, num_rows + 1};

    if (is_general || is_spd) {
        exec->run(isai::make_generate_general_inverse(
            lend(to_invert), lend(inverted), excess_block_ptrs.get_data(),
            excess_row_ptrs_full.get_data(), is_spd));
    } else {
        exec->run(isai::make_generate_tri_inverse(
            lend(to_invert), lend(inverted), excess_block_ptrs.get_data(),
            excess_row_ptrs_full.get_data(), is_lower));
    }

    auto total_excess_dim =
        exec->copy_val_to_host(excess_block_ptrs.get_const_data() + num_rows);
    auto excess_lim = excess_limit == 0 ? total_excess_dim : excess_limit;
    // if we had long rows:
    if (total_excess_dim > 0) {
        bool done = false;
        size_type block = 0;
        while (true) {
            // build the excess sparse triangular system
            size_type excess_dim = 0;
            size_type excess_start = block;
            const auto block_offset = exec->copy_val_to_host(
                excess_block_ptrs.get_const_data() + block);
            const auto nnz_offset = exec->copy_val_to_host(
                excess_row_ptrs_full.get_const_data() + block);
            while (excess_dim < excess_lim && block < num_rows) {
                block++;
                excess_dim = exec->copy_val_to_host(
                                 excess_block_ptrs.get_const_data() + block) -
                             block_offset;
            }
            if (excess_dim == 0) break;
            auto excess_nnz =
                exec->copy_val_to_host(excess_row_ptrs_full.get_const_data() +
                                       block) -
                nnz_offset;
            auto excess_system =
                Csr::create(exec, dim<2>(excess_dim, excess_dim), excess_nnz);
            excess_system->set_strategy(
                std::make_shared<typename Csr::classical>());
            auto excess_rhs = Dense::create(exec, dim<2>(excess_dim, 1));
            auto excess_solution = Dense::create(exec, dim<2>(excess_dim, 1));
            exec->run(isai::make_generate_excess_system(
                lend(to_invert), lend(inverted),
                excess_block_ptrs.get_const_data(),
                excess_row_ptrs_full.get_const_data(), lend(excess_system),
                lend(excess_rhs), excess_start, block));
            // solve it after transposing
            auto system_copy = Csr::create(exec->get_master());
            system_copy->copy_from(excess_system.get());
            auto rhs_copy = Dense::create(exec->get_master());
            rhs_copy->copy_from(excess_rhs.get());
            std::shared_ptr<LinOpFactory> excess_solver_factory;
            if (parameters_.excess_solver_factory) {
                excess_solver_factory =
                    share(parameters_.excess_solver_factory);
                excess_solution->copy_from(excess_rhs.get());
            } else if (is_general || is_spd) {
                excess_solver_factory =
                    Gmres::build()
                        .with_preconditioner(
                            bj::build().with_max_block_size(32u).on(exec))
                        .with_criteria(
                            gko::stop::Iteration::build()
                                .with_max_iters(excess_dim)
                                .on(exec),
                            gko::stop::RelativeResidualNorm<ValueType>::build()
                                .with_tolerance(remove_complex<ValueType>{1e-6})
                                .on(exec))
                        .on(exec);
                excess_solution->copy_from(excess_rhs.get());
            } else if (is_lower) {
                excess_solver_factory = UpperTrs::build().on(exec);
            } else {
                excess_solver_factory = LowerTrs::build().on(exec);
            }
            excess_solver_factory->generate(share(excess_system->transpose()))
                ->apply(lend(excess_rhs), lend(excess_solution));
            if (is_spd) {
                exec->run(isai::make_scale_excess_solution(
                    excess_block_ptrs.get_const_data(), lend(excess_solution),
                    excess_start, block));
            }
            // and copy the results back to the original ISAI
            exec->run(isai::make_scatter_excess_solution(
                excess_block_ptrs.get_const_data(), lend(excess_solution),
                lend(inverted), excess_start, block));
        }
    }

    approximate_inverse_ = std::move(inverted);
}


template <isai_type IsaiType, typename ValueType, typename IndexType,
          typename StorageType>
std::unique_ptr<LinOp>
Isai<IsaiType, ValueType, IndexType, StorageType>::transpose() const
{
    auto exec = this->get_executor();
    auto is_spd = IsaiType == isai_type::spd;
    if (is_spd) {
        return this->clone();
    }

    std::unique_ptr<transposed_type> transp{new transposed_type{exec}};
    transp->set_size(gko::transpose(this->get_size()));
    auto csr_transp =
        share(as<Csr>(as<Csr>(this->get_approximate_inverse())->transpose()));
    auto ell_transp = convert_matrix_formats<Ell>(csr_transp);
    transp->approximate_inverse_ = share(ell_transp);

    return std::move(transp);
}


template <isai_type IsaiType, typename ValueType, typename IndexType,
          typename StorageType>
std::unique_ptr<LinOp>
Isai<IsaiType, ValueType, IndexType, StorageType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto is_spd = IsaiType == isai_type::spd;
    if (is_spd) {
        return this->clone();
    }

    std::unique_ptr<transposed_type> transp{new transposed_type{exec}};
    transp->set_size(gko::transpose(this->get_size()));
    auto csr_transp = share(
        as<Csr>(as<Csr>(this->get_approximate_inverse())->conj_transpose()));
    auto ell_transp = convert_matrix_formats<Ell>(csr_transp);
    transp->approximate_inverse_ = share(ell_transp);

    return std::move(transp);
}


#define GKO_DECLARE_LOWER_ISAI(ValueType, IndexType, StorageType) \
    class Isai<isai_type::lower, ValueType, IndexType, StorageType>
GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_STORAGE_TYPE(GKO_DECLARE_LOWER_ISAI);

#define GKO_DECLARE_UPPER_ISAI(ValueType, IndexType, StorageType) \
    class Isai<isai_type::upper, ValueType, IndexType, StorageType>
GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_STORAGE_TYPE(GKO_DECLARE_UPPER_ISAI);

#define GKO_DECLARE_GENERAL_ISAI(ValueType, IndexType, StorageType) \
    class Isai<isai_type::general, ValueType, IndexType, StorageType>
GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_STORAGE_TYPE(GKO_DECLARE_GENERAL_ISAI);

#define GKO_DECLARE_SPD_ISAI(ValueType, IndexType, StorageType) \
    class Isai<isai_type::spd, ValueType, IndexType, StorageType>
GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_STORAGE_TYPE(GKO_DECLARE_SPD_ISAI);


}  // namespace preconditioner
}  // namespace gko
