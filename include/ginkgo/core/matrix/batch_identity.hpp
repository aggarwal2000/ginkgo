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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_


#include <ginkgo/core/base/batch_lin_op.hpp>


namespace gko {
namespace matrix {


/**
 * This class is a utility which efficiently implements the identity matrix (a
 * linear operator which maps each vector to itself).
 *
 * Thus, objects of the BatchIdentity class always represent a square matrix,
 * and don't require any storage for their values. The apply method is
 * implemented as a simple copy (or a linear combination).
 *
 * @note This class is useful when composing it with other operators. For
 *       example, it can be used instead of a preconditioner in Krylov solvers,
 *       if one wants to run a "plain" solver, without using a preconditioner.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup identity
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchIdentity : public EnableBatchLinOp<BatchIdentity<ValueType>>,
                      public EnableCreateMethod<BatchIdentity<ValueType>>,
                      public BatchTransposable {
    friend class EnablePolymorphicObject<BatchIdentity, BatchLinOp>;
    friend class EnableCreateMethod<BatchIdentity>;

public:
    using EnableBatchLinOp<BatchIdentity>::convert_to;
    using EnableBatchLinOp<BatchIdentity>::move_to;

    using value_type = ValueType;
    using transposed_type = BatchIdentity<ValueType>;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;


protected:
    /**
     * Creates an empty BatchIdentity matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit BatchIdentity(std::shared_ptr<const Executor> exec)
        : EnableBatchLinOp<BatchIdentity>(exec)
    {}

    /**
     * Creates an BatchIdentity matrix of the specified size.
     *
     * @param size  size of the matrix (must be square)
     */
    BatchIdentity(std::shared_ptr<const Executor> exec, size_type num_batches,
                  dim<2> size)
        : EnableBatchLinOp<BatchIdentity>(
              exec, std::vector<dim<2>>(num_batches, size))
    {
        GKO_ASSERT_IS_BATCH_SQUARE_MATRIX(this);
    }

    /**
     * Creates an BatchIdentity matrix of the specified size.
     *
     * @param size  size of the matrix
     */
    BatchIdentity(std::shared_ptr<const Executor> exec, size_type num_batches,
                  size_type size)
        : EnableBatchLinOp<BatchIdentity>(
              exec, std::vector<dim<2>>(num_batches, dim<2>{size}))
    {}

    void apply_impl(const BatchLinOp *b, BatchLinOp *x) const override;

    void apply_impl(const BatchLinOp *alpha, const BatchLinOp *b,
                    const BatchLinOp *beta, BatchLinOp *x) const override;
};


/**
 * This factory is a utility which can be used to generate BatchIdentity
 * operators.
 *
 * The factory will generate the BatchIdentity matrix with the same dimension as
 * the passed in operator. It will throw an exception if the operator is not
 * square.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchIdentityFactory
    : public EnablePolymorphicObject<BatchIdentityFactory<ValueType>,
                                     BatchLinOpFactory> {
    friend class EnablePolymorphicObject<BatchIdentityFactory,
                                         BatchLinOpFactory>;

public:
    using value_type = ValueType;

    /**
     * Creates a new BatchIdentity factory.
     *
     * @param exec  the executor where the BatchIdentity operator will be stored
     *
     * @return a unique pointer to the newly created factory
     */
    static std::unique_ptr<BatchIdentityFactory> create(
        std::shared_ptr<const Executor> exec)
    {
        return std::unique_ptr<BatchIdentityFactory>(
            new BatchIdentityFactory(std::move(exec)));
    }

protected:
    std::unique_ptr<BatchLinOp> generate_impl(
        std::shared_ptr<const BatchLinOp> base) const override;

    BatchIdentityFactory(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<BatchIdentityFactory, BatchLinOpFactory>(exec)
    {}
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_IDENTITY_HPP_
