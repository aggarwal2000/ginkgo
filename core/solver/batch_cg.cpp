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

#include <ginkgo/core/solver/batch_cg.hpp>


#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/matrix/batch_csr_kernels.hpp"
#include "core/matrix/batch_dense_kernels.hpp"
#include "core/solver/batch_cg_kernels.hpp"
#include "core/solver/batch_solver.ipp"


namespace gko {
namespace solver {
namespace batch_cg {


GKO_REGISTER_OPERATION(apply, batch_cg::apply);


}  // namespace batch_cg


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchCg<ValueType>::transpose() const
{
    return build()
        .with_preconditioner(parameters_.preconditioner)
        .with_max_iterations(parameters_.max_iterations)
        .with_residual_tol(parameters_.residual_tol)
        .with_tolerance_type(parameters_.tolerance_type)
        .on(this->get_executor())
        ->generate(share(
            as<BatchTransposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<BatchLinOp> BatchCg<ValueType>::conj_transpose() const
{
    return build()
        .with_preconditioner(parameters_.preconditioner)
        .with_max_iterations(parameters_.max_iterations)
        .with_residual_tol(parameters_.residual_tol)
        .with_tolerance_type(parameters_.tolerance_type)
        .on(this->get_executor())
        ->generate(share(as<BatchTransposable>(this->get_system_matrix())
                             ->conj_transpose()));
}


template <typename ValueType>
void BatchCg<ValueType>::solver_apply(const BatchLinOp* const mtx,
                                      const BatchLinOp* const b,
                                      BatchLinOp* const x,
                                      BatchInfo* const info) const
{
    using Dense = matrix::BatchDense<ValueType>;
    const kernels::batch_cg::BatchCgOptions<remove_complex<ValueType>> opts{
        parameters_.preconditioner, parameters_.max_iterations,
        parameters_.residual_tol, parameters_.tolerance_type};
    auto exec = this->get_executor();
    exec->run(batch_cg::make_apply(
        opts, mtx, as<const Dense>(b), as<Dense>(x),
        *as<log::BatchLogData<ValueType>>(info->logdata.get())));
}


#define GKO_DECLARE_BATCH_CG(_type) class BatchCg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG);


#define GKO_DECLARE_BATCH_CG_APPLY_FUNCTION(_type)                            \
    void EnableBatchSolver<BatchCg<_type>, BatchLinOp>::apply_impl(           \
        const BatchLinOp* b, BatchLinOp* x) const;                            \
    template void EnableBatchSolver<BatchCg<_type>, BatchLinOp>::apply_impl(  \
        const BatchLinOp* alpha, const BatchLinOp* b, const BatchLinOp* beta, \
        BatchLinOp* x) const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CG_APPLY_FUNCTION);


}  // namespace solver
}  // namespace gko
