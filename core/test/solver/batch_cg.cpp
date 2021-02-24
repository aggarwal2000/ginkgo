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

#include <ginkgo/core/solver/batch_cg.hpp>


#include <typeinfo>


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class BatchCg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::BatchDense<value_type>;
    using Solver = gko::solver::BatchCg<value_type>;

    BatchCg()
        : exec(gko::ReferenceExecutor::create()),
          mtx(gko::batch_initialize<Mtx>(
              {{{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}},
               {{4, -1.0, 0.0}, {-1.0, 4, -1.0}, {0.0, -1.0, 4}}},
              exec)),
          batch_cg_factory(
              Solver::build()
                  .with_criteria(
                      gko::stop::Iteration::build().with_max_iters(3u).on(exec),
                      gko::stop::ResidualNorm<value_type>::build()
                          .with_reduction_factor(gko::remove_complex<T>{1e-6})
                          .on(exec))
                  .on(exec)),
          solver(batch_cg_factory->generate(mtx))
    {}

    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> batch_cg_factory;
    std::unique_ptr<gko::BatchLinOp> solver;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_num_batches(), m2->get_num_batches());
        for (gko::size_type b = 0; b < m1->get_num_batches(); ++b) {
            ASSERT_EQ(m1->get_sizes()[b][0], m2->get_sizes()[b][0]);
            ASSERT_EQ(m1->get_sizes()[b][1], m2->get_sizes()[b][1]);
            for (gko::size_type i = 0; i < m1->get_sizes()[b][0]; ++i) {
                for (gko::size_type j = 0; j < m2->get_sizes()[b][1]; ++j) {
                    EXPECT_EQ(m1->at(b, i, j), m2->at(b, i, j));
                }
            }
        }
    }
};

TYPED_TEST_SUITE(BatchCg, gko::test::ValueTypes);


TYPED_TEST(BatchCg, BatchCgFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->batch_cg_factory->get_executor(), this->exec);
}


TYPED_TEST(BatchCg, BatchCgFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_sizes()[0], gko::dim<2>(3, 3));
    ASSERT_EQ(this->solver->get_sizes()[1], gko::dim<2>(3, 3));
    auto batch_cg_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(batch_cg_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(batch_cg_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchCg, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batch_cg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_sizes()[0], gko::dim<2>(3, 3));
    ASSERT_EQ(copy->get_sizes()[0], gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(BatchCg, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->batch_cg_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_sizes()[0], gko::dim<2>(3, 3));
    ASSERT_EQ(copy->get_sizes()[0], gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(BatchCg, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_sizes()[0], gko::dim<2>(3, 3));
    ASSERT_EQ(clone->get_sizes()[0], gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(BatchCg, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_sizes().size(), 0);
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(BatchCg, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(BatchCg, CanSetPreconditionerGenerator)
{
    using Solver = typename TestFixture::Solver;
    using value_type = typename TestFixture::value_type;
    auto batch_cg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec),
                gko::stop::ResidualNorm<value_type>::build()
                    .with_reduction_factor(
                        gko::remove_complex<value_type>(1e-6))
                    .on(this->exec))
            .with_preconditioner(
                Solver::build()
                    .with_criteria(
                        gko::stop::Iteration::build().with_max_iters(3u).on(
                            this->exec))
                    .on(this->exec))
            .on(this->exec);
    auto solver = batch_cg_factory->generate(this->mtx);
    auto precond = dynamic_cast<const gko::solver::BatchCg<value_type> *>(
        static_cast<gko::solver::BatchCg<value_type> *>(solver.get())
            ->get_preconditioner()
            .get());

    ASSERT_NE(precond, nullptr);
    ASSERT_EQ(precond->get_sizes()[0], gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_sizes()[1], gko::dim<2>(3, 3));
    ASSERT_EQ(precond->get_system_matrix(), this->mtx);
}


TYPED_TEST(BatchCg, CanSetPreconditionerInFactory)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> batch_cg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto batch_cg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(batch_cg_precond)
            .on(this->exec);
    auto solver = batch_cg_factory->generate(this->mtx);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), batch_cg_precond.get());
}


TYPED_TEST(BatchCg, ThrowsOnWrongPreconditionerInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> wrong_sized_mtx = Mtx::create(
        this->exec,
        std::vector<gko::dim<2>>{gko::dim<2>{2, 2}, gko::dim<2>{2, 2}});
    std::shared_ptr<Solver> batch_cg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(wrong_sized_mtx);

    auto batch_cg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .with_generated_preconditioner(batch_cg_precond)
            .on(this->exec);

    ASSERT_THROW(batch_cg_factory->generate(this->mtx), gko::DimensionMismatch);
}


TYPED_TEST(BatchCg, ThrowsOnRectangularMatrixInFactory)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Mtx> rectangular_mtx =
        Mtx::create(this->exec, std::vector<gko::dim<2>>(2, gko::dim<2>{1, 2}));

    ASSERT_THROW(this->batch_cg_factory->generate(rectangular_mtx),
                 gko::DimensionMismatch);
}


TYPED_TEST(BatchCg, CanSetPreconditioner)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> batch_cg_precond =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec)
            ->generate(this->mtx);

    auto batch_cg_factory =
        Solver::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(3u).on(this->exec))
            .on(this->exec);
    auto solver = batch_cg_factory->generate(this->mtx);
    solver->set_preconditioner(batch_cg_precond);
    auto precond = solver->get_preconditioner();

    ASSERT_NE(precond.get(), nullptr);
    ASSERT_EQ(precond.get(), batch_cg_precond.get());
}


}  // namespace
