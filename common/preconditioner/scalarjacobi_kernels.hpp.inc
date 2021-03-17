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


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_kernel(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs, const ValueType *__restrict__ vals,
    ValueType *__restrict__ inv)
{
    constexpr auto warp_size = config::warp_size;
    const auto row_idx = thread::get_subwarp_id_flat<warp_size>();
    auto warp_tile =
        group::tiled_partition<warp_size>(group::this_thread_block());


    if (row_idx < num_rows) {
        auto temp = zero<ValueType>();

        for (auto i = warp_tile.thread_rank() + row_ptrs[row_idx];
             i < row_ptrs[row_idx + 1]; i += warp_size) {
            if (col_idxs[i] == row_idx) {
                temp = vals[i];

                inv[row_idx] = temp;
            }
        }

        __syncwarp();

        if (warp_tile.thread_rank() == 0) {
            if (inv[row_idx] != zero<ValueType>()) {
                inv[row_idx] = one<ValueType>() / inv[row_idx];
            } else {
                printf(
                    "\n Scalar jacobi preconditioner generation fails due to 0 "
                    "at diagonal position, %d, %s ",
                    __LINE__, __FILE__);
            }
        }
    }
}


template <typename ValueType>
__global__ __launch_bounds__(default_block_size) void simple_apply_kernel(
    size_type num_rows, size_type num_cols, size_type stride_source,
    const ValueType *__restrict__ source, size_type stride_result,
    ValueType *__restrict__ result, const ValueType *__restrict__ inv)
{
    const auto global_id = thread::get_thread_id_flat();

    const auto row = global_id / num_cols;
    const auto col = global_id % num_cols;

    const auto scale_factor = inv[row];

    if (row < num_rows) {
        result[row * stride_result + col] =
            source[row * stride_source + col] * scale_factor;
    }
}


}  // namespace kernel