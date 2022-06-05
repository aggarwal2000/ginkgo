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

/**
 * Batch exact ilu0 preconditioner class.
 *
 */
template <typename ValueType>
class batch_exact_ilu0 final {
public:
    using value_type = ValueType;

    /**
     * @param factored_mat_batch   Factored (csr format) matrix (-combined L and
     * U) that was externally generated.
     * @param diag_ptrs_csr  Pointers to diagonal entries of factored matrix
     */
    batch_exact_ilu0(
        const gko::batch_csr::UniformBatch<const ValueType>& factored_mat_batch,
        const int* __restrict__ const diag_ptrs_csr)
        : factored_mat_batch_{factored_mat_batch}, diag_ptrs_csr_{diag_ptrs_csr}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    __host__ __device__ static constexpr int dynamic_work_size(int nrows,
                                                               int nnz)
    {
        return nrows;
    }

    /**
     * Complete the precond generation process.
     *
     * @param mat  Matrix for which to build an ILU-type preconditioner.
     */
    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        auto factored_mat_entry_ =
            gko::batch::batch_entry(factored_mat_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        auto factored_mat_entry_ =
            gko::batch::batch_entry(factored_mat_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        auto factored_mat_entry_ =
            gko::batch::batch_entry(factored_mat_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void apply(
        const int num_rows, const ValueType* const __restrict__ r,
        volatile ValueType* const __restrict__ z) const
    {
        printf("\n Apply- exact ilu0 precond apply - yet to be implemented");
        /*
        // vec_hat = precond * vec
        // => L * U  * vec_hat = vec
        // => L * y = vec , find y , and then U * vec_hat = y, find vec_hat

        // sparse triangular solves

        // if we want to use the busy waiting while loop approach, then the
        // num_rows should be <= threadblock size, else there is possibility of
        // a deadlock!
        assert(num_rows <= blockDim.x);
        // TODO: For upper trsv, use thread 0 for the bottommost row, this way
        // we could avoid :  assert(num_rows <= blockDim.x), as there won't be a
        // possibility of deadlock then!

        volatile ValueType* __restrict__ temp_vec = work_;

        for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
            temp_vec[i] = 1.8 / 0;  // TODO: find a better way to deal with
                                    // this!
            z[i] = 1.1 / 0;
        }

        __syncthreads();

        legacy_sparse_lower_trsv_combined(r, temp_vec);

        __syncthreads();

        legacy_sparse_upper_trsv_combined(temp_vec, z);
        */
    }


private:
    const gko::batch_csr::UniformBatch<const value_type> factored_mat_batch_;
    const int* __restrict__ const diag_ptrs_csr_;
    gko::batch_csr::BatchEntry<const value_type> factored_mat_entry_;
    ValueType* __restrict__ work_;

    /*
    __device__ __forceinline__ void legacy_sparse_lower_trsv_combined(
    const ValueType* const __restrict__ vec,
    volatile ValueType* __restrict__ const temp_vec)
    {
        const int num_rows = factored_values_entry_.num_rows;
        const int* const __restrict__ row_ptrs = factored_mat_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = factored_mat_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
    factored_mat_entry_.values;

        const int row_index = threadIdx.x;

        if (row_index >= num_rows) {
            return;
        }

        ValueType sum = 0;

        const int start = row_ptrs[row_index];
        const int end = diag_ptrs_csr_[row_index];

        int i = start;

        bool completed = false;

        while (!completed) {
            const int col_index = col_idxs[i];

            if (i < end && isfinite(temp_vec[col_index])) {
                sum += factored_values_entry[i] * temp_vec[col_index];
                i++;
            }

            if (i == end) {
                temp_vec[row_index] =
                    (vec[row_index] - sum) /
                    1;  // 1 in place of factored_values_entry[end]

                completed = true;
            }
        }
    }


    __device__ __forceinline__ void legacy_sparse_upper_trsv_combined(
        volatile const ValueType* const __restrict__ temp_vec,
        volatile ValueType* const __restrict__ vec_hat)
    {
        const int num_rows = factored_values_entry_.num_rows;
        const int* const __restrict__ row_ptrs = factored_mat_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = factored_mat_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
    factored_mat_entry_.values;

        const int row_index = threadIdx.x;

        if (row_index >= num_rows) {
            return;
        }

        ValueType sum = 0;

        const int start = diag_ptrs_csr_[row_index];
        const int end = row_ptrs[row_index + 1] - 1;
        int i = end;

        bool completed = false;

        while (!completed) {
            const int col_index = col_idxs[i];

            if (i > start && isfinite(vec_hat[col_index])) {
                sum += factored_values_entry[i] * vec_hat[col_index];
                i--;
            }

            if (i == start) {
                vec_hat[row_index] =
                    (temp_vec[row_index] - sum) / factored_values_entry[start];
                completed = true;
            }
        }
    }
    */
};