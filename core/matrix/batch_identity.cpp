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

#include <ginkgo/core/matrix/batch_identity.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
void BatchIdentity<ValueType>::apply_impl(const BatchLinOp *b,
                                          BatchLinOp *x) const
{
    x->copy_from(b);
}


template <typename ValueType>
void BatchIdentity<ValueType>::apply_impl(const BatchLinOp *alpha,
                                          const BatchLinOp *b,
                                          const BatchLinOp *beta,
                                          BatchLinOp *x) const
{
    auto dense_x = as<BatchDense<ValueType>>(x);
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, b);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchIdentityFactory<ValueType>::generate_impl(
    std::shared_ptr<const BatchLinOp> base) const
{
    GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(base, batch_transpose(base->get_sizes()));
    return BatchIdentity<ValueType>::create(
        this->get_executor(), base->get_num_batches(), base->get_sizes()[0][0]);
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchIdentity<ValueType>::transpose() const
{
    return this->clone();
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchIdentity<ValueType>::conj_transpose() const
{
    return this->clone();
}


#define GKO_DECLARE_BATCH_IDENTITY_MATRIX(_type) class BatchIdentity<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDENTITY_MATRIX);
#define GKO_DECLARE_BATCH_IDENTITY_FACTORY(_type) \
    class BatchIdentityFactory<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDENTITY_FACTORY);


}  // namespace matrix
}  // namespace gko
