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

#ifndef GKO_CORE_PRECONDITIONER_SCALARJACOBI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_SCALARJACOBI_KERNELS_HPP_


#include <ginkgo/core/preconditioner/scalarjacobi.hpp>


#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_SCALARJACOBI_GENERATE_KERNEL(ValueType)           \
    void generate(std::shared_ptr<const DefaultExecutor> exec,        \
                  const matrix::Csr<ValueType, int32> *system_matrix, \
                  Array<ValueType> &inv_eles)

#define GKO_DECLARE_SCALARJACOBI_APPLY_KERNEL(ValueType)    \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               const Array<ValueType> &inv_eles,            \
               const matrix::Dense<ValueType> *alpha,       \
               const matrix::Dense<ValueType> *b,           \
               const matrix::Dense<ValueType> *beta,        \
               matrix::Dense<ValueType> *x)

#define GKO_DECLARE_SCALARJACOBI_SIMPLE_APPLY_KERNEL(ValueType)    \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const Array<ValueType> &inv_eles,            \
                      const matrix::Dense<ValueType> *b,           \
                      matrix::Dense<ValueType> *x)


#define GKO_DECLARE_ALL_AS_TEMPLATES                     \
    template <typename ValueType>                        \
    GKO_DECLARE_SCALARJACOBI_GENERATE_KERNEL(ValueType); \
    template <typename ValueType>                        \
    GKO_DECLARE_SCALARJACOBI_APPLY_KERNEL(ValueType);    \
    template <typename ValueType>                        \
    GKO_DECLARE_SCALARJACOBI_SIMPLE_APPLY_KERNEL(ValueType)


namespace omp {
namespace scalarjacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace scalarjacobi
}  // namespace omp


namespace cuda {
namespace scalarjacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace scalarjacobi
}  // namespace cuda


namespace reference {
namespace scalarjacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace scalarjacobi
}  // namespace reference


namespace hip {
namespace scalarjacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace scalarjacobi
}  // namespace hip


namespace dpcpp {
namespace scalarjacobi {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace scalarjacobi
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_SCALARJACOBI_KERNELS_HPP_
