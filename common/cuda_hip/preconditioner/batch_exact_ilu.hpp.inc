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


namespace {


// load and store functions by written by Tobias to deal with volatile pointer
// compilation errors taken from file: (cuda/solver/common_trs_kernels.cuh)
// TODO: Put these in an appropriate file
template <typename ValueType, typename IndexType>
__device__ __forceinline__
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType>
    load(const ValueType* values, IndexType index)
{
    const volatile ValueType* val = values + index;
    return *val;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ std::enable_if_t<
    std::is_floating_point<ValueType>::value, thrust::complex<ValueType>>
load(const thrust::complex<ValueType>* values, IndexType index)
{
    auto real = reinterpret_cast<const ValueType*>(values);
    auto imag = real + 1;
    return {load(real, 2 * index), load(imag, 2 * index)};
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void store(
    ValueType* values, IndexType index,
    std::enable_if_t<std::is_floating_point<ValueType>::value, ValueType> value)
{
    volatile ValueType* val = values + index;
    *val = value;
}

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void store(thrust::complex<ValueType>* values,
                                      IndexType index,
                                      thrust::complex<ValueType> value)
{
    auto real = reinterpret_cast<ValueType*>(values);
    auto imag = real + 1;
    store(real, 2 * index, value.real());
    store(imag, 2 * index, value.imag());
}

}  // namespace


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
        factored_mat_entry_ =
            gko::batch::batch_entry(factored_mat_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        factored_mat_entry_ =
            gko::batch::batch_entry(factored_mat_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        factored_mat_entry_ =
            gko::batch::batch_entry(factored_mat_batch_, batch_id);
        work_ = work;
    }


    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const __restrict__ r,
                                          ValueType* const __restrict__ z) const
    {
        const int* const __restrict__ row_ptrs = factored_mat_entry_.row_ptrs;
        const int* const __restrict__ col_idxs = factored_mat_entry_.col_idxs;
        const ValueType* const __restrict__ factored_values_entry =
            factored_mat_entry_.values;

        // z = precond * r  ==> find z: L * U * z = r
        /*
        if (threadIdx.x == 0) {
            // L * work_ = r
            for (int row_idx = 0; row_idx < num_rows; row_idx++) {
                ValueType sum{0};

                for (int i = row_ptrs[row_idx]; i < diag_ptrs_csr_[row_idx];
                     i++) {
                    sum += factored_values_entry[i] * work_[col_idxs[i]];
                }
                work_[row_idx] = r[row_idx] - sum;
            }

            // U * z = work_
            for (int row_idx = num_rows - 1; row_idx >= 0; row_idx--) {
                ValueType sum{0};

                for (int i = row_ptrs[row_idx + 1] - 1;
                     i > diag_ptrs_csr_[row_idx]; i--) {
                    sum += factored_values_entry[i] * z[col_idxs[i]];
                }

                z[row_idx] = (work_[row_idx] - sum) /
                             factored_values_entry[diag_ptrs_csr_[row_idx]];
            }
        }*/

        // TODO: Add kernels for volta architecture and its successors (which
        // make use of independent thread scheduling)

        for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
            work_[i] =
                nan<ValueType>();  // TODO: find a better way to deal with this!
            z[i] = nan<ValueType>();
        }

        // L * work_ = r
        int row_index = threadIdx.x;
        if (row_index < num_rows) {
            ValueType sum = 0;

            const int start = row_ptrs[row_index];
            const int end = diag_ptrs_csr_[row_index];

            int i = start;
            bool completed = false;

            while (!completed) {
                const int col_index = col_idxs[i];

                if (i < end && is_finite(load(work_, col_index))) {
                    sum += factored_values_entry[i] * load(work_, col_index);
                    i++;
                }

                if (i == end) {
                    ValueType val =
                        (r[row_index] - sum) /
                        1;  // 1 in place of factored_values_entry[end]
                    store(work_, row_index, val);
                    completed = true;
                }
            }
        }

        __syncthreads();

        // U * z = work_
        row_index = num_rows - 1 - static_cast<int>(threadIdx.x);
        if (row_index >= 0) {
            ValueType sum = 0;

            const int start = diag_ptrs_csr_[row_index];
            const int end = row_ptrs[row_index + 1] - 1;
            int i = end;

            bool completed = false;

            while (!completed) {
                const int col_index = col_idxs[i];

                if (i > start && is_finite(load(z, col_index))) {
                    sum += factored_values_entry[i] * load(z, col_index);
                    i--;
                }

                if (i == start) {
                    ValueType val =
                        (work_[row_index] - sum) / factored_values_entry[start];
                    store(z, row_index, val);
                    completed = true;
                }
            }
        }
    }


private:
    const gko::batch_csr::UniformBatch<const value_type> factored_mat_batch_;
    const int* __restrict__ const diag_ptrs_csr_;
    gko::batch_csr::BatchEntry<const value_type> factored_mat_entry_;
    ValueType* __restrict__ work_;
};