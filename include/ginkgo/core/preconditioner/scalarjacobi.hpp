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

#ifndef GKO_PUBLIC_CORE_PRECONDITIONER_SCALARJACOBI_HPP_
#define GKO_PUBLIC_CORE_PRECONDITIONER_SCALARJACOBI_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
/**
 * @brief The Preconditioner namespace.
 *
 * @ingroup precond
 */
namespace preconditioner {


/**
 * A Scalarjacobi preconditioner is a diagonal linear operator, obtained
 * by inverting the diagonal elements of the source operator.
 *
 * The Scalarjacobi class implements the inversion of the diagonal elements
 * and stores the inverses explicitly in an array.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup scalarjacobi
 * @ingroup precond
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class Scalarjacobi : public EnableLinOp<Scalarjacobi<ValueType>>,
                     public ConvertibleTo<matrix::Dense<ValueType>>,
                     public WritableToMatrixData<ValueType, int32>,
                     public WritableToMatrixData<ValueType, int64>,
                     public Transposable {
    friend class EnableLinOp<Scalarjacobi>;
    friend class EnablePolymorphicObject<Scalarjacobi, LinOp>;

public:
    using EnableLinOp<Scalarjacobi>::convert_to;
    using EnableLinOp<Scalarjacobi>::move_to;
    using value_type = ValueType;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using transposed_type = Scalarjacobi<ValueType>;


    /**
     * Returns the pointer to the memory used for storing the inverse elements.
     *
     * @return the pointer to the memory used for storing the inverse elements
     *
     * @internal
     */
    const value_type *get_inv_eles() const noexcept
    {
        return inv_eles_.get_const_data();
    }


    /**
     * Returns the number of elements explicitly stored in the preconditioner
     * matrix.
     *
     * @return the number of elements explicitly stored in the preconditioner
     * matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return inv_eles_.get_num_elems();
    }

    void convert_to(matrix::Dense<value_type> *result) const override;

    void move_to(matrix::Dense<value_type> *result) override;


    void write(mat_data &data) const override;

    void write(mat_data32 &data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    GKO_CREATE_FACTORY_PARAMETERS(parameters, Factory){

    };
    GKO_ENABLE_LIN_OP_FACTORY(Scalarjacobi, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);


protected:
    /**
     * Creates an empty Scalarjacobi preconditioner.
     *
     * @param exec  the executor this object is assigned to
     */
    explicit Scalarjacobi(std::shared_ptr<const Executor> exec)
        : EnableLinOp<Scalarjacobi>(exec), inv_eles_(exec)
    {}


    /**
     * Creates a Scalarjacobi preconditioner from a matrix using a
     * Scalarjacobi::Factory.
     *
     * @param factory  the factory to use to create the preconditoner
     * @param system_matrix  the matrix this preconditioner should be created
     *                       from
     */
    explicit Scalarjacobi(const Factory *factory,
                          std::shared_ptr<const LinOp> system_matrix)
        : EnableLinOp<Scalarjacobi>(factory->get_executor(),
                                    system_matrix->get_size()),
          parameters_{factory->get_parameters()},
          inv_eles_(factory->get_executor(), system_matrix->get_size()[0])
    {
        this->generate_scalar_jacobi(lend(system_matrix));
    }


    /**
     * Generates the preconditoner.
     *
     * @param system_matrix  the source matrix used to generate the
     *                       preconditioner
     */
    void generate_scalar_jacobi(const LinOp *system_matrix);

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    Array<value_type> inv_eles_;
};


}  // namespace preconditioner
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_PRECONDITIONER_SCALARJACOBI_HPP_
