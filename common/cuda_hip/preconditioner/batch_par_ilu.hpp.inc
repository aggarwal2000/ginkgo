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
 * Batch parilu0 preconditioner class.
 *
 */
template <typename ValueType>
class batch_parilu0 final {
public:
    using value_type = ValueType;

    /**
     * @param l_batch  Lower triangular factor that was externally generated.
     * @param u_batch  Upper triangular factor that was externally generated.
     */
    batch_parilu0(const gko::batch_csr::UniformBatch<const ValueType>& l_batch,
                  const gko::batch_csr::UniformBatch<const ValueType>& u_batch)
        : l_batch_{l_batch}, u_batch_{u_batch}
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
        auto l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        auto u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        auto l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        auto u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        auto l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        auto u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const __restrict__ r,
                                          ValueType* const __restrict__ z) const
    {
        const int* const __restrict__ L_row_ptrs = l_entry_.row_ptrs;
        const int* const __restrict__ L_col_idxs = l_entry_.col_idxs;
        const ValueType* const __restrict__ L_values = l_entry_.values;

        const int* const __restrict__ U_row_ptrs = u_entry_.row_ptrs;
        const int* const __restrict__ U_col_idxs = u_entry_.col_idxs;
        const ValueType* const __restrict__ U_values = u_entry_.values;

        if (threadIdx.x == 0) {
            printf("TODO- TRSV");

            // L * work_ = r
            // Note: L is in sorted order
            for (int row_idx = 0; row_idx < num_rows; row_idx++) {
                printf("\n row idx: %d", row_idx);
                const int row_st = L_row_ptrs[row_idx];
                const int row_end = L_row_ptrs[row_idx + 1];

                ValueType sum{0};

                for (int i = row_st; i < row_end - 1; i++) {
                    printf("\n i: %d", i);
                    const int col_idx = L_col_idxs[i];
                    sum += L_values[i] * work_[col_idx];
                }

                work_[row_idx] = (r[row_idx] - sum) / L_values[row_end - 1];
                printf("\n work at row idx: %d ", row_idx);
            }

            printf("Lower TRSV done");

            // U * z = work_
            // note: U is in sorted order
            for (int row_idx = num_rows - 1; row_idx >= 0; row_idx--) {
                const int row_st = U_row_ptrs[row_idx];
                const int row_end = U_row_ptrs[row_idx + 1];

                ValueType sum{0};

                for (int i = row_end - 1; i > row_st; i--) {
                    const int col_idx = U_col_idxs[i];
                    sum += U_values[i] * z[col_idx];
                }

                z[row_idx] = (work_[row_idx] - sum) / U_values[row_st];
            }

            printf("Upper TRSV done");
        }
    }

    /*
    __device__ __forceinline__ void apply(
        const int num_rows, const ValueType* const __restrict__ r,
        volatile ValueType* const __restrict__ z) const
    {
       //  if(threadIdx.x == 0)
       // printf("\n Apply- parilu0 precond apply - yet to be implemented");

         // vec_hat = precond * vec
         // => L * U  * vec_hat = vec
         // => L * y = vec , find y , and then U * vec_hat = y, find vec_hat

         //sparse triangular solves

         //if we want to use the busy waiting while loop approach, then the
         // num_rows should be <= threadblock size, else there is possibility of
    a
         // deadlock !
         assert(num_rows <= blockDim.x);
         //TODO: For upper trsv, use thread 0 for the bottommost row, this way
       //  we could avoid :  assert(num_rows <= blockDim.x), as there won't be a
       //  possibility of deadlock then!

         volatile ValueType* const __restrict__ temp_vec = work_;

         for(int i = threadIdx.x ; i < num_rows; i += blockDim.x)
         {
             temp_vec[i] = 1.8/0; //TODO: find a better way to deal with this!
             z[i] = 1.3/0;

         }
         __syncthreads();

         legacy_sparse_lower_trsv(r,temp_vec);
         __syncthreads();

         legacy_sparse_upper_trsv(temp_vec, z);

    }*/


private:
    const gko::batch_csr::UniformBatch<const value_type> l_batch_;
    const gko::batch_csr::UniformBatch<const value_type> u_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_entry_;
    gko::batch_csr::BatchEntry<const value_type> u_entry_;
    ValueType* __restrict__ work_;

    /*
    __device__ __forceinline__ void legacy_sparse_lower_trsv(const ValueType*
    const __restrict__ input_vec, volatile ValueType* const __restrict__
    output_vec)
    {
        const int num_rows = l_entry_.num_rows;
        const int* const __restrict__ L_row_ptrs = l_entry_.row_ptrs;
        const int* const __restrict__ L_col_idxs = l_entry_.col_idxs;
        const ValueType* const __restrict__ L_values = l_entry_.values;

        const int row_index = threadIdx.x;

        if(row_index >= num_rows)
        {
            return;
        }

        ValueType sum = zero<ValueType>();

        const int start = L_row_ptrs[row_index];
        const int end = L_row_ptrs[row_index + 1] - 1;
        int i = start;

        bool completed = false;

        while(!completed)
        {
            const int col_index = L_col_idxs[i];

            if( i < end  &&  isfinite(output_vec[col_index]))
            {
                sum += L_values[i] * output_vec[col_index];
                i++;
            }

            if(i == end)
            {
                output_vec[row_index] = (input_vec[row_index] -
    sum)/L_values[end]; completed = true;
            }
        }
    }

    __device__ __forceinline__ void legacy_sparse_upper_trsv(volatile const
    ValueType* const __restrict__ input_vec, volatile ValueType* const
    __restrict__ output_vec)
    {
        const int num_rows = u_entry_.num_rows;
        const int* const __restrict__ U_row_ptrs = u_entry_.row_ptrs;
        const int* const __restrict__ U_col_idxs = u_entry_.col_idxs;
        const ValueType* const __restrict__ U_values = u_entry_.values;

        const int row_index = threadIdx.x;

        if(row_index >= num_rows)
        {
            return;
        }

        ValueType sum = zero<ValueType>();

        const int start = U_row_ptrs[row_index];
        const int end = U_row_ptrs[row_index + 1]  - 1;
        int i = end;

        bool completed = false;

        while(!completed )
        {
            const int col_index = U_col_idxs[i];

            if( i > start && isfinite(output_vec[col_index]))
            {
                sum += U_values[i] * output_vec[col_index];
                i--;
            }

            if(i == start)
            {
                output_vec[row_index] = (input_vec[row_index] -
    sum)/U_values[start]; completed = true;
            }

        }

    }*/
};