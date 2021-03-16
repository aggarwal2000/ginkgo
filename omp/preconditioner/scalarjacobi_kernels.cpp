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

#include "core/preconditioner/scalarjacobi_kernels.hpp"


#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <vector>


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/extended_float.hpp"
#include "omp/components/matrix_operations.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Scalarjacobi preconditioner namespace.
 *
 * @ingroup scalarjacobi
 */
namespace scalarjacobi {


template <typename ValueType>
void generate(std::shared_ptr<const OmpExecutor> exec,
              const matrix::Csr<ValueType, int32> *system_matrix,
              Array<ValueType> &inv_eles) GKO_NOT_IMPLEMENTED;
// {
//    auto temp = zero<ValueType>();

//    #pragma omp parallel for
//    for(size_t row_id  = 0; row_id < system_matrix->get_size()[0]; row_id ++)
//    {
//        temp = zero<ValueType>();
//        for(size_type i = system_matrix->get_const_row_ptrs[row_id]; i <
//        system_matrix->get_const_row_ptrs[row_id + 1]; i++)
//        {
//            if(system_matrix->get_const_col_idxs[i] == row_id)
//            {
//                temp = system_matrix->get_const_values[i] ;
//                break;
//            }
//        }

//        if(temp !=  zero<ValueType>())
//        {
//             inv_eles[row_id] = one<ValueType>()/temp;
//        }
//        else
//        {
//            printf("Scalar jacobi preconditioner can't be generated, as there
//            is a 0 at diagonal position, error: %d , %d ", __LINE__,
//            __FILE__);
//        }

//    }
// }


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCALARJACOBI_GENERATE_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const Array<ValueType> &inv_eles,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta,
           matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCALARJACOBI_APPLY_KERNEL);


template <typename ValueType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const Array<ValueType> &inv_eles,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x) GKO_NOT_IMPLEMENTED;
// {
//    #pragma omp parallel for
//    for(size_type i = 0; i < b->get_size()[0] ; i++)
//    {
//        auto scale_factor = inv_eles[i];

//        #pragma omp parallel for
//        for(size_type j = 0; j < b->get_size()[1]; j++)
//        {
//            x->at(i,j) = x->at(i,j)*scale_factor;
//        }
//    }
// }

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_SCALARJACOBI_SIMPLE_APPLY_KERNEL);


}  // namespace scalarjacobi
}  // namespace omp
}  // namespace kernels
}  // namespace gko
