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

#ifndef GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
#define GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_


#include "core/matrix/batch_struct.hpp"
#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {


template <typename ValueType>
class batch_ilu_isai final {
public:
    using value_type = ValueType;


    /**
     * @param l_left_inv_batch  lower triangular factor isai that was externally
     * generated.
     * @param u_left_inv_batch  upper triangular factor isai that was externally
     * generated.
     * @param mult_inv_batch    u_left_inv * l_left_inv
     * @param is_inv_factors_spgemm
     */
    batch_ilu_isai(
        const gko::batch_csr::UniformBatch<const ValueType>& l_left_inv_batch,
        const gko::batch_csr::UniformBatch<const ValueType>& u_left_inv_batch,
        const gko::batch_csr::UniformBatch<const ValueType>& mult_inv_batch,
        const bool is_inv_factors_spgemm)
        : l_left_inv_batch_{l_left_inv_batch},
          u_left_inv_batch_{u_left_inv_batch},
          mult_inv_batch_{mult_inv_batch},
          is_inv_factors_spgemm_{is_inv_factors_spgemm}
    {}


    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application) // used when is_inv_factors_spgemm =
     * false
     */
    static constexpr int dynamic_work_size(int nrows, int nnz) { return nrows; }


    /**
     * Complete the precond generation process.
     *
     * @param mat  Matrix for which to build an ILU-ISAI preconditioner.
     */
    void generate(size_type batch_id,
                  const gko::batch_csr::BatchEntry<const ValueType>&,
                  ValueType* const __restrict__ work)
    {
        auto l_left_inv_entry_ =
            gko::batch::batch_entry(l_left_inv_batch_, batch_id);
        auto u_left_inv_entry_ =
            gko::batch::batch_entry(u_left_inv_batch_, batch_id);
        auto mult_inv_entry_ =
            gko::batch::batch_entry(mult_inv_batch_, batch_id);  // maybe null
        work_ = work;
    }

    void generate(size_type batch_id,
                  const gko::batch_ell::BatchEntry<const ValueType>&,
                  ValueType* const __restrict__ work)
    {
        auto l_left_inv_entry_ =
            gko::batch::batch_entry(l_left_inv_batch_, batch_id);
        auto u_left_inv_entry_ =
            gko::batch::batch_entry(u_left_inv_batch_, batch_id);
        auto mult_inv_entry_ =
            gko::batch::batch_entry(mult_inv_batch_, batch_id);  // maybe null
        work_ = work;
    }

    void generate(size_type batch_id,
                  const gko::batch_dense::BatchEntry<const ValueType>&,
                  ValueType* const __restrict__ work)
    {
        auto l_left_inv_entry_ =
            gko::batch::batch_entry(l_left_inv_batch_, batch_id);
        auto u_left_inv_entry_ =
            gko::batch::batch_entry(u_left_inv_batch_, batch_id);
        auto mult_inv_entry_ =
            gko::batch::batch_entry(mult_inv_batch_, batch_id);  // maybe null
        work_ = work;
    }


    void apply(const gko::batch_dense::BatchEntry<const ValueType>& r,
               const gko::batch_dense::BatchEntry<ValueType>& z) const
        GKO_NOT_IMPLEMENTED;


private:
    const gko::batch_csr::UniformBatch<const value_type> l_left_inv_batch_;
    const gko::batch_csr::UniformBatch<const value_type> u_left_inv_batch_;
    const gko::batch_csr::UniformBatch<const value_type> mult_inv_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_left_inv_entry_;
    gko::batch_csr::BatchEntry<const value_type> u_left_inv_entry_;
    gko::batch_csr::BatchEntry<const value_type> mult_inv_entry_;
    const bool is_inv_factors_spgemm_;
    ValueType* __restrict__ work_;
};


}  // namespace host
}  // namespace kernels
}  // namespace gko

#endif  // GKO_REFERENCE_PRECONDITIONER_BATCH_ILU_ISAI_HPP_
