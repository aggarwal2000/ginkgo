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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {

/**
 * Method to use for factorization of the system matrix in a given sparsity
 * pattern.
 */
enum class batch_ilu_factorization_type { exact, par_ilu };


/**
 * A batch of ILU-ISAI preconditioners for a batch of matrices.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup ilu
 * @ingroup precond
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchIluIsai
    : public EnableBatchLinOp<BatchIluIsai<ValueType, IndexType>>,
      public BatchTransposable {
    friend class EnableBatchLinOp<BatchIluIsai>;
    friend class EnablePolymorphicObject<BatchIluIsai, BatchLinOp>;

public:
    using EnableBatchLinOp<BatchIluIsai>::convert_to;
    using EnableBatchLinOp<BatchIluIsai>::move_to;
    using value_type = ValueType;
    using index_type = IndexType;
    using matrix_type = matrix::BatchCsr<ValueType, IndexType>;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory)
    {
        /**
         * The `system_matrix`, which will be given to this factory, must be
         * sorted (first by row, then by column) in order for the algorithm
         * to work. If it is known that the matrix will be sorted, this
         * parameter can be set to `true` to skip the sorting (therefore,
         * shortening the runtime).
         * However, if it is unknown or if the matrix is known to be not sorted,
         * it must remain `false`, otherwise, this factorization might be
         * incorrect.
         */
        bool GKO_FACTORY_PARAMETER_SCALAR(skip_sorting, false);


        /**
         * Number of sweeps for parILU0 generation (in case parilu factorization
         * is used)
         *
         */
        int GKO_FACTORY_PARAMETER_SCALAR(par_ilu_num_sweeps, 10);


        /**
         * Factorization algorithm to use.
         */
        batch_ilu_factorization_type GKO_FACTORY_PARAMETER_SCALAR(
            ilu_factorization_type, batch_ilu_factorization_type::exact);

        /**
         * Sparisty pattern for lower triangular factor's isai
         *
         */
        int GKO_FACTORY_PARAMETER_SCALAR(lower_factor_isai_sparsity_power, 1);

        /**
         * Sparisty pattern for upper triangular factor's isai
         *
         */
        int GKO_FACTORY_PARAMETER_SCALAR(upper_factor_isai_sparsity_power, 1);


        bool GKO_FACTORY_PARAMETER_SCALAR(perform_inv_factors_batch_spgemm,
                                          false);
    };
    GKO_ENABLE_BATCH_LIN_OP_FACTORY(BatchIluIsai, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

    std::unique_ptr<BatchLinOp> transpose() const;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_lower_factor_left_approx_inverse() const
    {
        return l_left_isai_;
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_upper_factor_left_approx_inverse() const
    {
        return u_left_isai_;
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_lower_factor() const
    {
        return l_factor_;
    }

    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_upper_factor() const
    {
        return u_factor_;
    }

    // u_left_isai_ * l_left_isai_
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>>
    get_const_u_left_isai_mult_l_left_isai() const
    {
        return mult_inv_;
    }

    bool get_is_inv_factors_batch_spgemm_performed() const
    {
        return parameters_.perform_inv_factors_batch_spgemm;
    }

protected:
    /**
     * Creates an empty Ilu preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit BatchIluIsai(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchIluIsai>(exec)
    {}

    /**
     * Creates a Ilu preconditioner from a matrix using a Ilu::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit BatchIluIsai(const Factory* factory,
                          std::shared_ptr<const BatchLinOp> system_matrix)
        : EnableBatchLinOp<BatchIluIsai>(
              factory->get_executor(),
              gko::transpose(system_matrix->get_size())),
          parameters_{factory->get_parameters()}

    {
        this->generate(lend(system_matrix));
    }

    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate(const BatchLinOp* system_matrix);

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override{};

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override{};

private:
    // Note: Make these two const to avoid cloning the isai precond's approx inv
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>> l_left_isai_;
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>> u_left_isai_;
    std::shared_ptr<matrix::BatchCsr<ValueType, IndexType>> mult_inv_;
    // Note: Make these two const to avoid cloning the parilu precond's factors
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>> l_factor_;
    std::shared_ptr<const matrix::BatchCsr<ValueType, IndexType>> u_factor_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
