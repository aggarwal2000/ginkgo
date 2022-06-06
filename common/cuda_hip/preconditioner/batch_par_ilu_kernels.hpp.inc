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
__device__ __forceinline__ void parilu0_sweep_for_batch_entry(
    const int nnz, const ValueType* const __restrict__ A_vals,
    ValueType* const __restrict__ L_vals, ValueType* const __restrict__ U_vals,
    const int* const __restrict__ dependencies,
    const int* const __restrict__ nz_ptrs)
{
    for (int el = threadIdx.x; el < nnz;
         el += blockDim.x)  // non-coalesced accesses, but data locality
    {
        ValueType diag_val{one<ValueType>()};

        const int st = nz_ptrs[el];
        const int end = nz_ptrs[el + 1] - 1;

        bool has_diag_dependency = (end + 1 - st) % 2 == 0 ? true : false;

        ValueType sum{zero<ValueType>()};

        for (int idx = st + 1; idx <= end - 1; idx += 2) {
            sum += L_vals[dependencies[idx]] * U_vals[dependencies[idx + 1]];
        }

        if (has_diag_dependency == true) {
            diag_val = U_vals[dependencies[end]];
        }

        ValueType to_write = (A_vals[el] - sum) / diag_val;

        if (has_diag_dependency == true) {
            L_vals[dependencies[st]] = to_write;
        } else {
            U_vals[dependencies[st]] = to_write;
        }
    }
}

template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void compute_parilu0_kernel(
    const size_type nbatch, const int nrows, const int nnz,
    const ValueType* const __restrict__ values, const int L_nnz,
    ValueType* const __restrict__ L_values, const int U_nnz,
    ValueType* const __restrict__ U_values, const int num_sweeps,
    const int* const __restrict__ dependencies,
    const int* const __restrict__ nz_ptrs)
{
    for (size_type batch_id = blockIdx.x; batch_id < nbatch;
         batch_id += gridDim.x) {
        extern __shared__ char local_mem_sh[];
        ValueType* const __restrict__ L_values_sh =
            reinterpret_cast<ValueType*>(local_mem_sh);
        ValueType* const __restrict__ U_values_sh = L_values_sh + L_nnz;

        single_copy(L_nnz, L_values + batch_id * L_nnz, L_values_sh);
        single_copy(U_nnz, U_values + batch_id * U_nnz, U_values_sh);
        __syncthreads();

        for (int sweep = 0; sweep < num_sweeps; sweep++) {
            parilu0_sweep_for_batch_entry(nnz, values + batch_id * nnz,
                                          L_values_sh, U_values_sh,
                                          dependencies, nz_ptrs);
            __syncthreads();
        }

        single_copy(L_nnz, L_values_sh, L_values + batch_id * L_nnz);
        single_copy(U_nnz, U_values_sh, U_values + batch_id * U_nnz);
    }
}