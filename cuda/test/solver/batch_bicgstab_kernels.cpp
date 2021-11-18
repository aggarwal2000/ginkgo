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

#include <ginkgo/core/solver/batch_bicgstab.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/log/batch_convergence.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch_test_utils.hpp"


namespace {


template <typename T>
class BatchBicgstab : public ::testing::Test {
protected:
    using value_type = T;
    using real_type = gko::remove_complex<value_type>;
    using solver_type = gko::solver::BatchBicgstab<T>;
    using Mtx = gko::matrix::BatchCsr<value_type, int>;
    using BDense = gko::matrix::BatchDense<value_type>;
    using RBDense = gko::matrix::BatchDense<real_type>;
    using Options =
        gko::kernels::batch_bicgstab::BatchBicgstabOptions<real_type>;
    using LogData = gko::log::BatchLogData<value_type>;

    BatchBicgstab()
        : exec(gko::ReferenceExecutor::create()),
          cuexec(gko::CudaExecutor::create(0, exec)),
          sys_1(gko::test::get_poisson_problem<T>(exec, 1, nbatch))
    {
        auto execp = cuexec;
        solve_fn = [execp](const Options opts, const Mtx* mtx, const BDense* b,
                           BDense* x, LogData& logdata) {
            gko::kernels::cuda::batch_bicgstab::apply<value_type>(
                execp, opts, mtx, b, x, logdata);
        };
        scale_mat = [execp](const BDense* const left, const BDense* const right,
                            Mtx* const mat, BDense* const b) {
            gko::kernels::cuda::batch_csr::pre_diag_scale_system<value_type>(
                execp, left, right, mat, b);
        };
        scale_vecs = [execp](const BDense* const scale, BDense* const mat) {
            gko::kernels::cuda::batch_dense::batch_scale<value_type>(
                execp, scale, mat);
        };
    }

    void TearDown()
    {
        if (cuexec != nullptr) {
            ASSERT_NO_THROW(cuexec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> exec;
    std::shared_ptr<const gko::CudaExecutor> cuexec;

    const real_type eps = r<value_type>::value;

    const size_t nbatch = 2;
    const int nrows = 3;
    const Options opts_1{gko::preconditioner::batch::type::none, 500,
                         static_cast<real_type>(1e3) * eps,
                         gko::stop::batch::ToleranceType::relative};
    gko::test::LinSys<T> sys_1;

    std::function<void(Options, const Mtx*, const BDense*, BDense*, LogData&)>
        solve_fn;
    std::function<void(const BDense*, const BDense*, Mtx*, BDense*)> scale_mat;
    std::function<void(const BDense*, BDense*)> scale_vecs;

    std::unique_ptr<typename solver_type::Factory> create_factory(
        std::shared_ptr<const gko::Executor> exec, const Options& opts)
    {
        return solver_type::build()
            .with_max_iterations(opts.max_its)
            .with_residual_tol(opts.residual_tol)
            .with_tolerance_type(opts.tol_type)
            .with_preconditioner(opts.preconditioner)
            .on(exec);
    }

    int single_iters_regression()
    {
        if (std::is_same<real_type, float>::value) {
            return 2;
        } else if (std::is_same<real_type, double>::value) {
            return 2;
        } else {
            return -1;
        }
    }
};

TYPED_TEST_SUITE(BatchBicgstab, gko::test::ValueTypes);


TYPED_TEST(BatchBicgstab, SolveIsEquivalentToReference)
{
    using value_type = typename TestFixture::value_type;
    using solver_type = gko::solver::BatchBicgstab<value_type>;
    using opts_type = typename TestFixture::Options;
    using mtx_type = typename TestFixture::Mtx;
    constexpr bool issingle =
        std::is_same<gko::remove_complex<value_type>, float>::value;
    const float solver_restol = issingle ? 50 * this->eps : this->eps;
    const opts_type opts{gko::preconditioner::batch::type::none, 500,
                         solver_restol,
                         gko::stop::batch::ToleranceType::relative};
    auto r_sys = gko::test::generate_solvable_batch_system<mtx_type>(
        this->exec, this->nbatch, 11, 1, false);
    auto r_factory = this->create_factory(this->exec, opts);
    const double iter_tol = 0.01;
    const double res_tol = 10 * r<value_type>::value;
    const double sol_tol = 10 * solver_restol;

    gko::test::compare_with_reference<value_type, solver_type>(
        this->cuexec, r_sys, r_factory.get(), iter_tol, res_tol, sol_tol);
}


TYPED_TEST(BatchBicgstab, StencilSystemLoggerIsCorrect)
{
    using value_type = typename TestFixture::value_type;
    using real_type = gko::remove_complex<value_type>;

    auto r_1 = gko::test::solve_poisson_uniform(
        this->cuexec, this->solve_fn, this->scale_mat, this->scale_vecs,
        this->opts_1, this->sys_1, 1);

    const int ref_iters = this->single_iters_regression();
    const int* const iter_array = r_1.logdata.iter_counts.get_const_data();
    const real_type* const res_log_array =
        r_1.logdata.res_norms->get_const_values();
    for (size_t i = 0; i < this->nbatch; i++) {
        std::cout << "Iterations are " << iter_array[i]
                  << ", logged norm = " << res_log_array[i] << std::endl;
        GKO_ASSERT((iter_array[i] <= ref_iters + 1) &&
                   (iter_array[i] >= ref_iters - 1));
        ASSERT_LE(res_log_array[i] / this->sys_1.bnorm->at(i, 0, 0),
                  this->opts_1.residual_tol);
        ASSERT_NEAR(res_log_array[i], r_1.resnorm->get_const_values()[i],
                    10 * this->eps);
    }
}


TYPED_TEST(BatchBicgstab, CoreSolvesSystemJacobi)
{
    using value_type = typename TestFixture::value_type;
    using Mtx = typename TestFixture::Mtx;
    using BDense = typename TestFixture::BDense;
    using Solver = gko::solver::BatchBicgstab<value_type>;
    auto useexec = this->cuexec;
    std::unique_ptr<typename Solver::Factory> batchbicgstab_factory =
        Solver::build()
            .with_max_iterations(100)
            .with_residual_tol(1e-6f)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .on(useexec);
    const int nrhs_1 = 1;
    const size_t nbatch = 3;
    const auto sys =
        gko::test::get_poisson_problem<value_type>(this->exec, nrhs_1, nbatch);
    auto rx =
        gko::batch_initialize<BDense>(nbatch, {0.0, 0.0, 0.0}, this->exec);
    std::unique_ptr<Mtx> mtx = Mtx::create(useexec);
    auto b = BDense::create(useexec);
    auto x = BDense::create(useexec);
    mtx->copy_from(gko::lend(sys.mtx));
    b->copy_from(gko::lend(sys.b));
    x->copy_from(gko::lend(rx));

    std::unique_ptr<Solver> solver =
        batchbicgstab_factory->generate(gko::give(mtx));
    solver->apply(b.get(), x.get());
    rx->copy_from(gko::lend(x));

    GKO_ASSERT_BATCH_MTX_NEAR(rx, sys.xex, 1e-5);
}


TYPED_TEST(BatchBicgstab, UnitScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.0, 1.0}, this->exec);
    auto factory = this->create_factory(this->cuexec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->cuexec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, this->eps);
}


TYPED_TEST(BatchBicgstab, GeneralScalingDoesNotChangeResult)
{
    using BDense = typename TestFixture::BDense;
    using Solver = typename TestFixture::solver_type;
    auto left_scale = gko::batch_initialize<BDense>(
        this->nbatch, {0.8, 0.9, 0.95}, this->exec);
    auto right_scale = gko::batch_initialize<BDense>(
        this->nbatch, {1.0, 1.5, 1.05}, this->exec);
    auto factory = this->create_factory(this->cuexec, this->opts_1);

    auto result = gko::test::solve_poisson_uniform_core<Solver>(
        this->cuexec, factory.get(), this->sys_1, 1, left_scale.get(),
        right_scale.get());

    GKO_ASSERT_BATCH_MTX_NEAR(result.x, this->sys_1.xex, this->eps);
}


TEST(BatchBicgstab, GoodScalingImprovesConvergence)
{
    using value_type = double;
    using real_type = gko::remove_complex<value_type>;
    using Solver = gko::solver::BatchBicgstab<value_type>;
    const auto eps = r<value_type>::value;
    auto exec = gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> cuexec =
        gko::CudaExecutor::create(0, exec);
    const size_t nbatch = 3;
    const int nrows = 100;
    const int nrhs = 1;
    auto factory =
        Solver::build()
            .with_max_iterations(10)
            .with_residual_tol(10 * eps)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::none)
            .on(cuexec);

    gko::test::test_solve_iterations_with_scaling<Solver>(cuexec, nbatch, nrows,
                                                          nrhs, factory.get());
}


TEST(BatchBicgstab, CanSolveWithoutScaling)
{
    using T = std::complex<float>;
    using RT = typename gko::remove_complex<T>;
    using Solver = gko::solver::BatchBicgstab<T>;
    const RT tol = 1e-5;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> exec =
        gko::CudaExecutor::create(0, refexec);
    const int maxits = 5000;
    auto batchbicgstab_factory =
        Solver::build()
            .with_max_iterations(maxits)
            .with_residual_tol(tol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(exec);
    const int nrows = 29;
    const size_t nbatch = 3;
    const int nrhs = 1;

    gko::test::test_solve<Solver>(exec, nbatch, nrows, nrhs, tol, maxits,
                                  batchbicgstab_factory.get(), 5);
}


TEST(BatchBicgstab, SolvesLargeCsrSystemEquivalentToReference)
{
    using value_type = double;
    using real_type = double;
    using mtx_type = gko::matrix::BatchCsr<value_type, int>;
    using solver_type = gko::solver::BatchBicgstab<value_type>;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> d_exec =
        gko::CudaExecutor::create(0, refexec);
    const float solver_restol = 1e-4;
    auto r_sys = gko::test::generate_solvable_batch_system<mtx_type>(
        refexec, 2, 990, 1, false);
    auto r_factory =
        solver_type::build()
            .with_max_iterations(500)
            .with_residual_tol(solver_restol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(refexec);
    const double iter_tol = 0.01;
    const double res_tol = 1e-9;
    const double sol_tol = 10 * solver_restol;

    gko::test::compare_with_reference<value_type, solver_type>(
        d_exec, r_sys, r_factory.get(), iter_tol, res_tol, sol_tol);
}


TEST(BatchBicgstab, SolvesLargeDenseSystemEquivalentToReference)
{
    using value_type = double;
    using real_type = double;
    using mtx_type = gko::matrix::BatchDense<value_type>;
    using solver_type = gko::solver::BatchBicgstab<value_type>;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> d_exec =
        gko::CudaExecutor::create(0, refexec);
    const float solver_restol = 1e-4;
    auto r_sys = gko::test::generate_solvable_batch_system<mtx_type>(
        refexec, 2, 33, 1, false);
    auto r_factory =
        solver_type::build()
            .with_max_iterations(500)
            .with_residual_tol(solver_restol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(refexec);
    const double iter_tol = 0.01;
    const double res_tol = 1e-9;
    const double sol_tol = 10 * solver_restol;

    gko::test::compare_with_reference<value_type, solver_type>(
        d_exec, r_sys, r_factory.get(), iter_tol, res_tol, sol_tol);
}


TEST(BatchBicgstab, SolvesLargeEllSystemEquivalentToReference)
{
    using value_type = double;
    using real_type = double;
    using mtx_type = gko::matrix::BatchEll<value_type, int>;
    using solver_type = gko::solver::BatchBicgstab<value_type>;
    std::shared_ptr<gko::ReferenceExecutor> refexec =
        gko::ReferenceExecutor::create();
    std::shared_ptr<const gko::CudaExecutor> d_exec =
        gko::CudaExecutor::create(0, refexec);
    const float solver_restol = 1e-4;
    auto r_sys = gko::test::generate_solvable_batch_system<mtx_type>(
        refexec, 2, 90, 1, false);
    auto r_factory =
        solver_type::build()
            .with_max_iterations(500)
            .with_residual_tol(solver_restol)
            .with_tolerance_type(gko::stop::batch::ToleranceType::relative)
            .with_preconditioner(gko::preconditioner::batch::type::jacobi)
            .on(refexec);
    const double iter_tol = 0.01;
    const double res_tol = 1e-9;
    const double sol_tol = 10 * solver_restol;

    gko::test::compare_with_reference<value_type, solver_type>(
        d_exec, r_sys, r_factory.get(), iter_tol, res_tol, sol_tol);
}

}  // namespace
