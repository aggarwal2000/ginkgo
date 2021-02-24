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


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchIdentity : public ::testing::Test {
protected:
    using value_type = T;
    using Id = gko::matrix::BatchIdentity<T>;
    using Vec = gko::matrix::BatchDense<T>;

    BatchIdentity() : exec(gko::ReferenceExecutor::create()) {}

    std::shared_ptr<const gko::Executor> exec;
};

TYPED_TEST_SUITE(BatchIdentity, gko::test::ValueTypes);


TYPED_TEST(BatchIdentity, CanBeEmpty)
{
    using Id = typename TestFixture::Id;
    auto empty = Id::create(this->exec);
    ASSERT_EQ(empty->get_sizes().size(), 0);
}


TYPED_TEST(BatchIdentity, CanBeConstructedWithSize)
{
    using Id = typename TestFixture::Id;
    auto identity = Id::create(this->exec, 2, 5);

    ASSERT_EQ(identity->get_num_batches(), 2);
    ASSERT_EQ(identity->get_sizes()[0], gko::dim<2>(5, 5));
    ASSERT_EQ(identity->get_sizes()[1], gko::dim<2>(5, 5));
}


TYPED_TEST(BatchIdentity, CanBeConstructedWithSquareSize)
{
    using Id = typename TestFixture::Id;
    auto identity = Id::create(this->exec, 2, gko::dim<2>(5, 5));

    ASSERT_EQ(identity->get_num_batches(), 2);
    ASSERT_EQ(identity->get_sizes()[0], gko::dim<2>(5, 5));
    ASSERT_EQ(identity->get_sizes()[1], gko::dim<2>(5, 5));
}


TYPED_TEST(BatchIdentity, FailsConstructionWithRectangularSize)
{
    using Id = typename TestFixture::Id;

    ASSERT_THROW(Id::create(this->exec, 2, gko::dim<2>(5, 4)),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchIdentity, AppliesToVector)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    auto identity = Id::create(this->exec, 2, 3);
    auto x = Vec::create(this->exec, std::vector<gko::dim<2>>{
                                         gko::dim<2>{3, 1}, gko::dim<2>{3, 1}});
    auto b = gko::batch_initialize<Vec>({{2.0, 1.0, 5.0}, {2.0, 1.0, 5.0}},
                                        this->exec);

    identity->apply(b.get(), x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x, l({{2.0, 1.0, 5.0}, {2.0, 1.0, 5.0}}), 0.0);
}


TYPED_TEST(BatchIdentity, AppliesToMultipleVectors)
{
    using Id = typename TestFixture::Id;
    using Vec = typename TestFixture::Vec;
    using T = typename TestFixture::value_type;
    auto identity = Id::create(this->exec, 2, 3);
    auto x = Vec::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{3, 2}, gko::dim<2>{3, 2}},
        std::vector<gko::size_type>{3, 3});
    auto b = gko::batch_initialize<Vec>(
        std::vector<gko::size_type>{3, 3},
        {{I<T>{2.0, 3.0}, I<T>{1.0, 2.0}, I<T>{5.0, -1.0}},
         {I<T>{2.0, -3.0}, I<T>{1.0, 2.0}, I<T>{-5.0, -1.0}}},
        this->exec);

    identity->apply(b.get(), x.get());

    GKO_ASSERT_BATCH_MTX_NEAR(x,
                              l({{{2.0, 3.0}, {1.0, 2.0}, {5.0, -1.0}},
                                 {{2.0, -3.0}, {1.0, 2.0}, {-5.0, -1.0}}}),
                              0.0);
}


template <typename T>
class BatchIdentityFactory : public ::testing::Test {
protected:
    using value_type = T;
};

TYPED_TEST_SUITE(BatchIdentityFactory, gko::test::ValueTypes);


TYPED_TEST(BatchIdentityFactory, CanGenerateBatchIdentityMatrix)
{
    auto exec = gko::ReferenceExecutor::create();
    auto id_factory =
        gko::matrix::BatchIdentityFactory<TypeParam>::create(exec);
    auto mtx = gko::matrix::BatchDense<TypeParam>::create(
        exec, std::vector<gko::dim<2>>{gko::dim<2>{5, 5}, gko::dim<2>{5, 5}});

    auto id = id_factory->generate(std::move(mtx));

    ASSERT_EQ(id->get_sizes()[0], gko::dim<2>(5, 5));
    ASSERT_EQ(id->get_sizes()[1], gko::dim<2>(5, 5));
}


TYPED_TEST(BatchIdentityFactory, FailsToGenerateRectangularBatchIdentityMatrix)
{
    auto exec = gko::ReferenceExecutor::create();
    auto id_factory =
        gko::matrix::BatchIdentityFactory<TypeParam>::create(exec);
    auto mtx = gko::matrix::BatchDense<TypeParam>::create(
        exec, std::vector<gko::dim<2>>{gko::dim<2>{5, 4}, gko::dim<2>{5, 4}});

    ASSERT_THROW(id_factory->generate(std::move(mtx)), gko::DimensionMismatch);
}


}  // namespace
