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


template <typename ValueType>
__device__ __forceinline__ void single_matvec_kernel_batch_isai(
    const gko::batch_csr::BatchEntry<const ValueType>& a,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ c)
{
    constexpr auto tile_size = config::warp_size;

    auto thread_block = group::this_thread_block();
    auto subwarp_grp = group::tiled_partition<tile_size>(thread_block);
    const auto subwarp_grp_id = static_cast<int>(threadIdx.x / tile_size);
    const int num_subwarp_grps_per_block = ceildiv(blockDim.x, tile_size);

    for (int row = subwarp_grp_id; row < a.num_rows;
         row += num_subwarp_grps_per_block) {
        const int row_start = a.row_ptrs[row];
        const int row_end = a.row_ptrs[row + 1];

        ValueType temp = zero<ValueType>();
        for (int i = subwarp_grp.thread_rank() + row_start; i < row_end;
             i += subwarp_grp.size()) {
            const int col = a.col_idxs[i];
            const ValueType val = a.values[i];
            temp += val * b[col];
        }

#pragma unroll
        for (int i = static_cast<int>(tile_size) / 2; i > 0; i /= 2) {
            temp += subwarp_grp.shfl_down(temp, i);
        }

        if (subwarp_grp.thread_rank() == 0) {
            c[row] = temp;
        }
    }
}


/**
 * Batch isai preconditioner class.
 *
 */
template <typename ValueType>
class batch_isai final {
public:
    using value_type = ValueType;

    /**
     * @param left_isai_batch  left isai that was externally
     * generated.
     */
    batch_isai(
        const gko::batch_csr::UniformBatch<const ValueType>& left_isai_batch)
        : left_isai_batch_{left_isai_batch}
    {}


    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    __host__ __device__ static constexpr int dynamic_work_size(int nrows,
                                                               int nnz)
    {
        return 0;
    }

    /**
     * Complete the precond generation process.
     *
     * @param mat  Matrix for which to build an ILU-ISAI preconditioner.
     */
    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const)
    {
        left_isai_entry_ = gko::batch::batch_entry(left_isai_batch_, batch_id);
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const)
    {
        left_isai_entry_ = gko::batch::batch_entry(left_isai_batch_, batch_id);
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&, ValueType* const)
    {
        left_isai_entry_ = gko::batch::batch_entry(left_isai_batch_, batch_id);
    }

    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const __restrict__ r,
                                          ValueType* const __restrict__ z) const
    {
        // vec_hat = precond * vec
        // vec_hat = A_approx_inv * vec

        // TODO: Include appropriate header
        // single_matvec_kernel(left_isai_entry_, r, z);
        single_matvec_kernel_batch_isai(left_isai_entry_, r, z);
    }


private:
    const gko::batch_csr::UniformBatch<const value_type> left_isai_batch_;
    gko::batch_csr::BatchEntry<const value_type> left_isai_entry_;
};