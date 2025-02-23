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

#include "core/solver/cb_gmres_kernels.hpp"


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/range.hpp>


#include "core/base/accessors.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The CB_GMRES solver namespace.
 *
 * @ingroup cb_gmres
 */
namespace cb_gmres {


template <typename ValueType>
void initialize_1(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status,
                  size_type krylov_dim) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType, typename Accessor3d>
void initialize_2(std::shared_ptr<const DpcppExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<remove_complex<ValueType>> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
                  Accessor3d krylov_bases,
                  matrix::Dense<ValueType> *next_krylov_basis,
                  Array<size_type> *final_iter_nums,
                  size_type krylov_dim) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(
    GKO_DECLARE_CB_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType, typename Accessor3d>
void step_1(std::shared_ptr<const DpcppExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<remove_complex<ValueType>> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            Accessor3d krylov_bases, matrix::Dense<ValueType> *hessenberg_iter,
            matrix::Dense<ValueType> *buffer_iter,
            matrix::Dense<remove_complex<ValueType>> *arnoldi_norm,
            size_type iter, Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status,
            Array<stopping_status> *reorth_status,
            Array<size_type> *num_reorth) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_STEP_1_KERNEL);


template <typename ValueType, typename ConstAccessor3d>
void step_2(std::shared_ptr<const DpcppExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            ConstAccessor3d krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(
    GKO_DECLARE_CB_GMRES_STEP_2_KERNEL);


}  // namespace cb_gmres
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
