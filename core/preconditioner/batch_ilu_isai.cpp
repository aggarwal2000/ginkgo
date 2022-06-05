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
#include <ginkgo/core/preconditioner/batch_ilu_isai.hpp>
#include <ginkgo/core/preconditioner/batch_isai.hpp>
#include <ginkgo/core/preconditioner/batch_par_ilu.hpp>

namespace gko {
namespace preconditioner {
namespace batch_ilu_isai {
namespace {


// TODO: Register necessary operations


}  // namespace
}  // namespace batch_ilu_isai


template <typename ValueType, typename IndexType>
void BatchIluIsai<ValueType, IndexType>::generate(
    const BatchLinOp* const system_matrix)
{
    using unbatch_type = matrix::Csr<ValueType, IndexType>;
    using batch_csr = matrix::BatchCsr<ValueType, IndexType>;
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

    std::shared_ptr<const matrix_type> sys_csr_smart(sys_csr);

    std::shared_ptr<const batch_csr> l_factor;
    std::shared_ptr<const batch_csr> u_factor;

    if (parameters_.ilu_factorization_type =
            batch_ilu_factorization_type::par_ilu) {
        auto parilu_precond =
            gko::preconditioner::BatchParIlu<ValueType, IndexType>::build()
                .with_skip_sorting(parameters_.skip_sorting)
                .with_num_sweeps(parameters_.par_ilu_num_sweeps)
                .on(exec)
                ->generate(sys_csr_smart);
        l_factor = std::shared_ptr<const batch_csr>(
            parilu_precond->get_const_lower_factor());
        u_factor = std::shared_ptr<const batch_csr>(
            parilu_precond->get_const_upper_factor());
    } else {
        auto exact_ilu_precond =
            gko::preconditioner::BatchExactIlu<ValueType, IndexType>::build()
                .with_skip_sorting(parameters_.skip_sorting)
                .on(exec)
                ->generate(sys_csr_smart);

        std::shared_ptr<const batch_csr> factored_mat =
            std::shared_ptr<const batch_csr>(
                exact_ilu_precond->get_const_factorized_mat());

        // TODO:Split factors
        GKO_NOT_IMPLEMENTED;
    }


    auto lower_isai_precond =
        gko::preconditioner::BatchIsai<ValueType, IndexType>::build()
            .with_sparsity_power(parameters_.lower_factor_isai_sparsity_power)
            .with_matrix_type_isai(batch_isai_sys_mat_type::lower)
            .with_skip_sorting(true)
            .on(exec)
            ->generate(l_factor);

    auto upper_isai_precond =
        gko::preconditioner::BatchIsai<ValueType, IndexType>::build()
            .with_sparsity_power(parameters_.upper_factor_isai_sparsity_power)
            .with_matrix_type_isai(batch_isai_sys_mat_type::upper)
            .with_skip_sorting(true)
            .on(exec)
            ->generate(u_factor);

    const batch_csr* lower_inv =
        lower_isai_precond->get_const_left_approximate_inverse();
    const batch_csr* upper_inv =
        upper_isai_precond->get_const_left_approximate_inverse();

    l_left_isai_ = std::shared_ptr<const batch_csr>(lower_inv);
    u_left_isai_ = std::shared_ptr<const batch_csr>(upper_inv);

    std::shared_ptr<batch_csr> multipilcation_inv;

    if (parameters_.perform_inv_factors_batch_spgemm) {
        GKO_NOT_IMPLEMENTED;
        // mult_inv_ = batch_csr::create(exec);
        // Note::mult_inv_ should have allocated memory to store solution (u_inv
        // * l_inv)
        // u_left_isai_->apply(l_left_isai_, mult_inv_);
    }
}

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchIluIsai<ValueType, IndexType>::transpose()
    const
{
    GKO_NOT_IMPLEMENTED;
}

template <typename ValueType, typename IndexType>
std::unique_ptr<BatchLinOp> BatchIluIsai<ValueType, IndexType>::conj_transpose()
    const
{
    GKO_NOT_IMPLEMENTED;
}


}  // namespace preconditioner
}  // namespace gko
