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

#include <ginkgo/core/preconditioner/scalarjacobi.hpp>


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/extended_float.hpp"
#include "core/base/utils.hpp"
#include "core/preconditioner/scalarjacobi_kernels.hpp"


namespace gko {
namespace preconditioner {
namespace scalarjacobi {


GKO_REGISTER_OPERATION(simple_apply, scalarjacobi::simple_apply);
GKO_REGISTER_OPERATION(apply, scalarjacobi::apply);
GKO_REGISTER_OPERATION(generate, scalarjacobi::generate);


}  // namespace scalarjacobi

template <typename ValueType>
void Scalarjacobi<ValueType>::generate_scalar_jacobi(const LinOp *system_matrix)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix);
    using csr_type = matrix::Csr<ValueType, int32>;
    const auto exec = this->get_executor();
    auto csr_mtx = convert_to_with_sorting<csr_type>(
        exec, system_matrix, true);  // replace with normal function (We don't
                                     // want to sort if unsorted matrix)

    exec->run(scalarjacobi::make_generate(csr_mtx.get(), inv_eles_));
}


template <typename ValueType>
void Scalarjacobi<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using dense = matrix::Dense<ValueType>;

    std::cout << "\n In appy impl " << std::endl;

    this->get_executor()->run(
        scalarjacobi::make_simple_apply(inv_eles_, as<dense>(b), as<dense>(x)));
}


template <typename ValueType>
void Scalarjacobi<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                         const LinOp *beta, LinOp *x) const
{
    using dense = matrix::Dense<ValueType>;
    this->get_executor()->run(
        scalarjacobi::make_apply(inv_eles_, as<dense>(alpha), as<dense>(b),
                                 as<dense>(beta), as<dense>(x)));
}


template <typename ValueType>
void Scalarjacobi<ValueType>::convert_to(matrix::Dense<ValueType> *result) const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void Scalarjacobi<ValueType>::move_to(matrix::Dense<ValueType> *result)
    GKO_NOT_IMPLEMENTED;

template <typename ValueType>
void Scalarjacobi<ValueType>::write(mat_data &data) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
void Scalarjacobi<ValueType>::write(mat_data32 &data) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
std::unique_ptr<LinOp> Scalarjacobi<ValueType>::transpose() const
    GKO_NOT_IMPLEMENTED;


template <typename ValueType>
std::unique_ptr<LinOp> Scalarjacobi<ValueType>::conj_transpose() const
    GKO_NOT_IMPLEMENTED;


#define GKO_DECLARE_SCALARJACOBI(ValueType) class Scalarjacobi<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCALARJACOBI);


}  // namespace preconditioner
}  // namespace gko
