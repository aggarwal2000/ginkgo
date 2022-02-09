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

#include <ginkgo/core/matrix/batch_csr.hpp>


#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/test/utils/batch.hpp"
#include "cuda/test/utils.hpp"


namespace {


class BatchCsr : public ::testing::Test {
protected:
    using value_type = float;
    using real_type = gko::remove_complex<value_type>;
    using Vec = gko::matrix::BatchDense<value_type>;
    using Mtx = gko::matrix::BatchCsr<value_type>;
    using ComplexVec = gko::matrix::BatchDense<std::complex<value_type>>;
    using ComplexMtx = gko::matrix::BatchCsr<std::complex<value_type>>;
    using Dense = gko::matrix::Dense<value_type>;

    BatchCsr() : mtx_size(10, gko::dim<2>(62, 47)), rand_engine(42) {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        ref = gko::ReferenceExecutor::create();
        cuda = gko::CudaExecutor::create(0, ref);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    template <typename MtxType>
    std::unique_ptr<MtxType> gen_mtx(size_t batch_size, int num_rows,
                                     int num_cols, int min_nnz_row)
    {
        using real_type = typename gko::remove_complex<value_type>;
        return gko::test::generate_uniform_batch_random_matrix<MtxType>(
            batch_size, num_rows, num_cols,
            std::uniform_int_distribution<>(min_nnz_row, num_cols),
            std::normal_distribution<real_type>(0.0, 1.0), rand_engine, false,
            ref);
    }

    void set_up_apply_data(int num_vectors = 1)
    {
        const size_t batch_size = mtx_size.get_num_batch_entries();
        const int nrows = mtx_size.at()[0];
        const int ncols = mtx_size.at()[1];
        mtx = gen_mtx<Mtx>(batch_size, nrows, ncols, 1);
        square_mtx = gen_mtx<Mtx>(batch_size, nrows, nrows, 1);
        expected = gen_mtx<Vec>(batch_size, nrows, num_vectors, 1);
        y = gen_mtx<Vec>(batch_size, ncols, num_vectors, 1);
        alpha = gko::batch_initialize<Vec>(batch_size, {2.0}, ref);
        beta = gko::batch_initialize<Vec>(batch_size, {-1.0}, ref);
        dmtx = Mtx::create(cuda);
        dmtx->copy_from(mtx.get());
        square_dmtx = Mtx::create(cuda);
        square_dmtx->copy_from(square_mtx.get());
        dresult = Vec::create(cuda);
        dresult->copy_from(expected.get());
        dy = Vec::create(cuda);
        dy->copy_from(y.get());
        dalpha = Vec::create(cuda);
        dalpha->copy_from(alpha.get());
        dbeta = Vec::create(cuda);
        dbeta->copy_from(beta.get());
    }

    void set_up_apply_complex_data()
    {
        const size_t batch_size = mtx_size.get_num_batch_entries();
        const int nrows = mtx_size.at()[0];
        const int ncols = mtx_size.at()[1];
        complex_mtx = ComplexMtx::create(ref);
        complex_mtx->copy_from(
            gen_mtx<ComplexMtx>(batch_size, nrows, ncols, 1));
        complex_dmtx = ComplexMtx::create(cuda);
        complex_dmtx->copy_from(complex_mtx.get());

        complex_b = gen_mtx<ComplexVec>(batch_size, nrows, 3, 1);
        dcomplex_b = ComplexVec::create(cuda);
        dcomplex_b->copy_from(complex_b.get());
        complex_y = gen_mtx<ComplexVec>(batch_size, nrows, 3, 1);
        dcomplex_y = ComplexVec::create(cuda);
        dcomplex_y->copy_from(complex_y.get());
        complex_x = gen_mtx<ComplexVec>(batch_size, ncols, 3, 1);
        dcomplex_x = ComplexVec::create(cuda);
        dcomplex_x->copy_from(complex_x.get());
        c_alpha = gko::batch_initialize<ComplexVec>(batch_size, {2.0}, ref);
        c_beta = gko::batch_initialize<ComplexVec>(batch_size, {-1.0}, ref);
        dc_alpha = ComplexVec::create(cuda);
        dc_alpha->copy_from(c_alpha.get());
        dc_beta = ComplexVec::create(cuda);
        dc_beta->copy_from(c_beta.get());
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<const gko::CudaExecutor> cuda;

    const gko::batch_dim<> mtx_size;
    std::ranlux48 rand_engine;

    std::unique_ptr<Mtx> mtx;
    std::unique_ptr<ComplexMtx> complex_mtx;
    std::unique_ptr<Mtx> square_mtx;
    std::unique_ptr<Vec> expected;
    std::unique_ptr<Vec> y;
    std::unique_ptr<Vec> alpha;
    std::unique_ptr<Vec> beta;
    std::unique_ptr<ComplexVec> complex_b;
    std::unique_ptr<ComplexVec> complex_x;
    std::unique_ptr<ComplexVec> complex_y;
    std::unique_ptr<ComplexVec> c_alpha;
    std::unique_ptr<ComplexVec> c_beta;

    std::unique_ptr<Mtx> dmtx;
    std::unique_ptr<ComplexMtx> complex_dmtx;
    std::unique_ptr<Mtx> square_dmtx;
    std::unique_ptr<Vec> dresult;
    std::unique_ptr<Vec> dy;
    std::unique_ptr<Vec> dalpha;
    std::unique_ptr<Vec> dbeta;
    std::unique_ptr<ComplexVec> dcomplex_b;
    std::unique_ptr<ComplexVec> dcomplex_x;
    std::unique_ptr<ComplexVec> dcomplex_y;
    std::unique_ptr<ComplexVec> dc_alpha;
    std::unique_ptr<ComplexVec> dc_beta;
};


TEST_F(BatchCsr, SimpleApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(y.get(), expected.get());
    dmtx->apply(dy.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(BatchCsr, SimpleComplexApplyIsEquivalentToRef)
{
    set_up_apply_complex_data();

    complex_mtx->apply(complex_x.get(), complex_y.get());
    complex_dmtx->apply(dcomplex_x.get(), dcomplex_y.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dcomplex_y, complex_y, r<value_type>::value);
}


TEST_F(BatchCsr, AdvancedApplyIsEquivalentToRef)
{
    set_up_apply_data();

    mtx->apply(alpha.get(), y.get(), beta.get(), expected.get());
    dmtx->apply(dalpha.get(), dy.get(), dbeta.get(), dresult.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dresult, expected, r<value_type>::value);
}


TEST_F(BatchCsr, AdvancedComplexApplyIsEquivalentToRef)
{
    set_up_apply_complex_data();

    complex_mtx->apply(c_alpha.get(), complex_x.get(), c_beta.get(),
                       complex_y.get());
    complex_dmtx->apply(dc_alpha.get(), dcomplex_x.get(), dc_beta.get(),
                        dcomplex_y.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dcomplex_y, complex_y, r<value_type>::value);
}


TEST_F(BatchCsr, PreDiagScaleSystemIsEquivalentToReference)
{
    using BDiag = gko::matrix::BatchDiagonal<value_type>;
    set_up_apply_data();
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const size_t nrows = mtx_size.at()[0];
    const size_t ncols = mtx_size.at()[1];
    const int nrhs = 3;
    auto ref_left_scale = gen_mtx<BDiag>(batch_size, nrows, nrows, nrows);
    auto ref_right_scale = gen_mtx<BDiag>(batch_size, ncols, ncols, ncols);
    auto ref_b = gen_mtx<Vec>(batch_size, nrows, nrhs, 2);
    auto d_left_scale = BDiag::create(cuda);
    d_left_scale->copy_from(ref_left_scale.get());
    auto d_right_scale = BDiag::create(cuda);
    d_right_scale->copy_from(ref_right_scale.get());
    auto d_b = Vec::create(cuda);
    d_b->copy_from(ref_b.get());

    gko::kernels::reference::batch_csr::pre_diag_transform_system(
        ref, ref_left_scale.get(), ref_right_scale.get(), mtx.get(),
        ref_b.get());
    gko::kernels::cuda::batch_csr::pre_diag_transform_system(
        cuda, d_left_scale.get(), d_right_scale.get(), dmtx.get(), d_b.get());

    GKO_ASSERT_BATCH_MTX_NEAR(ref_b, d_b, 0.001 * r<value_type>::value);
    GKO_ASSERT_BATCH_MTX_NEAR(mtx, dmtx, 0.001 * r<value_type>::value);
}


TEST_F(BatchCsr, ConvertToBatchDenseIsEquivalentToReference)
{
    using Dense = gko::matrix::BatchDense<value_type>;
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gen_mtx<Mtx>(batch_size, nrows, ncols, nrows / 10);
    auto cmtx = Mtx::create(cuda);
    cmtx->copy_from(mtx.get());
    auto dense = Dense::create(ref, mtx_size);
    auto cdense = Dense::create(cuda, mtx_size);

    mtx->convert_to(dense.get());
    cmtx->convert_to(cdense.get());

    GKO_ASSERT_BATCH_MTX_NEAR(dense, cdense, 0.0);
}


TEST_F(BatchCsr, DetectsMissingDiagonalEntry)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gen_mtx<Mtx>(batch_size, nrows, ncols, nrows / 10);
    gko::test::remove_diagonal_from_row(mtx.get(), nrows / 2);
    auto dmtx = Mtx::create(cuda);
    dmtx->copy_from(mtx.get());
    bool all_diags = false;

    gko::kernels::cuda::batch_csr::check_diagonal_entries_exist(
        cuda, dmtx.get(), all_diags);

    ASSERT_FALSE(all_diags);
}


TEST_F(BatchCsr, DetectsPresenceOfAllDiagonalEntries)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batch_size, nrows, ncols,
        std::uniform_int_distribution<>(ncols / 10, ncols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    auto dmtx = Mtx::create(cuda);
    dmtx->copy_from(mtx.get());
    bool all_diags = false;

    gko::kernels::cuda::batch_csr::check_diagonal_entries_exist(
        cuda, dmtx.get(), all_diags);

    ASSERT_TRUE(all_diags);
}


TEST_F(BatchCsr, AddScaleIdentityIsEquivalentToReference)
{
    const size_t batch_size = mtx_size.get_num_batch_entries();
    const int nrows = mtx_size.at()[0];
    const int ncols = mtx_size.at()[1];
    auto mtx = gko::test::generate_uniform_batch_random_matrix<Mtx>(
        batch_size, nrows, ncols,
        std::uniform_int_distribution<>(ncols / 10, ncols),
        std::normal_distribution<>(-1.0, 1.0), rand_engine, true, ref);
    set_up_apply_data();
    auto dmtx = Mtx::create(cuda);
    dmtx->copy_from(mtx.get());

    mtx->add_scaled_identity(alpha.get(), beta.get());
    dmtx->add_scaled_identity(dalpha.get(), dbeta.get());

    GKO_ASSERT_BATCH_MTX_NEAR(mtx, dmtx, r<double>::value);
}


}  // namespace
