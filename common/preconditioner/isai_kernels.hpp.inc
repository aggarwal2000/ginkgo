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


/**
 * @internal
 *
 * This kernel supports at most `subwarp_size` (< `warp_size`) elements per row.
 * If there are more elements, they are simply ignored. Only the first
 * `subwarp_size` elements are considered both for the values and for the
 * sparsity pattern.
 */
template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType, typename Callable>
__forceinline__ __device__ void generic_generate(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values,
    const IndexType *__restrict__ i_row_ptrs,
    const IndexType *__restrict__ i_col_idxs, ValueType *__restrict__ i_values,
    IndexType *__restrict__ excess_rhs_sizes,
    IndexType *__restrict__ excess_nnz, Callable trs_solve)
{
    static_assert(subwarp_size >= row_size_limit, "incompatible subwarp_size");
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    if (row >= num_rows) {
        return;
    }

    const auto i_row_begin = i_row_ptrs[row];
    const auto i_row_size = i_row_ptrs[row + 1] - i_row_begin;

    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int local_id = subwarp.thread_rank();

    if (i_row_size > subwarp_size) {
        // defer long rows: store their nnz and number of matches
        IndexType count{};
        for (IndexType nz = 0; nz < i_row_size; ++nz) {
            auto col = i_col_idxs[i_row_begin + nz];
            auto m_row_begin = m_row_ptrs[col];
            auto m_row_size = m_row_ptrs[col + 1] - m_row_begin;
            // extract the sparse submatrix consisting of the entries whose
            // columns/rows match column indices from this row
            group_match<subwarp_size>(
                m_col_idxs + m_row_begin, m_row_size, i_col_idxs + i_row_begin,
                i_row_size, subwarp,
                [&](IndexType, IndexType, IndexType,
                    config::lane_mask_type matchmask,
                    bool) { count += popcnt(matchmask); });
        }
        // store the dim and nnz of this sparse block
        if (local_id == 0) {
            excess_rhs_sizes[row] = i_row_size;
            excess_nnz[row] = count;
        }
    } else {
        // handle short rows directly: no excess
        if (local_id == 0) {
            excess_rhs_sizes[row] = 0;
            excess_nnz[row] = 0;
        }

        // subwarp_size^2 storage per subwarp
        __shared__ UninitializedArray<ValueType, subwarp_size * subwarp_size *
                                                     subwarps_per_block>
            storage;

        auto trisystem_ptr = storage + (threadIdx.x / subwarp_size) *
                                           subwarp_size * subwarp_size;
        // row-major accessor
        auto trisystem = [&](IndexType row, IndexType col) -> ValueType & {
            return trisystem_ptr[row * subwarp_size + col];
        };

#pragma unroll
        for (int i = 0; i < subwarp_size; ++i) {
            trisystem(i, local_id) = zero<ValueType>();
        }

        subwarp.sync();

        for (IndexType nz = 0; nz < i_row_size; ++nz) {
            auto col = i_col_idxs[i_row_begin + nz];
            auto m_row_begin = m_row_ptrs[col];
            auto m_row_size = m_row_ptrs[col + 1] - m_row_begin;
            // extract the dense submatrix consisting of the entries whose
            // columns/rows match column indices from this row
            group_match<subwarp_size>(
                m_col_idxs + m_row_begin, m_row_size, i_col_idxs + i_row_begin,
                i_row_size, subwarp,
                [&](IndexType, IndexType m_idx, IndexType i_idx,
                    config::lane_mask_type, bool valid) {
                    if (valid) {
                        trisystem(nz, i_idx) = m_values[m_row_begin + m_idx];
                    }
                });
        }

        subwarp.sync();

        // Now, read a full col of `trisystem` into local registers, which will
        // be row elements after this (implicit) transpose
        ValueType local_row[subwarp_size];
#pragma unroll
        for (int i = 0; i < subwarp_size; ++i) {
            local_row[i] = trisystem(i, local_id);
        }

        const auto rhs = trs_solve(i_row_size, local_row, subwarp);

        // Write back:
        if (local_id < i_row_size) {
            const auto idx = i_row_begin + local_id;
            if (is_finite(rhs)) {
                i_values[idx] = rhs;
            } else {
                i_values[idx] = i_col_idxs[idx] == row ? one<ValueType>()
                                                       : zero<ValueType>();
            }
        }
    }
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_l_inverse(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values,
    const IndexType *__restrict__ i_row_ptrs,
    const IndexType *__restrict__ i_col_idxs, ValueType *__restrict__ i_values,
    IndexType *__restrict__ excess_rhs_sizes,
    IndexType *__restrict__ excess_nnz)
{
    auto trs_solve = [](IndexType num_elems,
                        const ValueType *__restrict__ local_row,
                        group::thread_block_tile<subwarp_size> &subwarp) {
        const int local_id = subwarp.thread_rank();
        ValueType rhs =
            local_id == num_elems - 1 ? one<ValueType>() : zero<ValueType>();
        // Solve Triangular system
        for (int d_col = num_elems - 1; d_col >= 0; --d_col) {
            const auto elem = local_row[d_col];
            if (d_col == local_id) {
                rhs /= elem;
            }

            const ValueType bot = subwarp.shfl(rhs, d_col);
            if (local_id < d_col) {
                rhs -= bot * elem;
            }
        }

        return rhs;
    };

    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, excess_rhs_sizes, excess_nnz, trs_solve);
}


template <int subwarp_size, int subwarps_per_block, typename ValueType,
          typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_u_inverse(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values,
    const IndexType *__restrict__ i_row_ptrs,
    const IndexType *__restrict__ i_col_idxs, ValueType *__restrict__ i_values,
    IndexType *__restrict__ excess_rhs_sizes,
    IndexType *__restrict__ excess_nnz)
{
    auto trs_solve = [](IndexType num_elems,
                        const ValueType *__restrict__ local_row,
                        group::thread_block_tile<subwarp_size> &subwarp) {
        const int local_id = subwarp.thread_rank();
        ValueType rhs = local_id == 0 ? one<ValueType>() : zero<ValueType>();
        // Solve Triangular system
        for (int d_col = 0; d_col < num_elems; ++d_col) {
            const auto elem = local_row[d_col];
            if (d_col == local_id) {
                rhs /= elem;
            }

            const ValueType top = subwarp.shfl(rhs, d_col);
            if (d_col < local_id) {
                rhs -= top * elem;
            }
        }

        return rhs;
    };

    generic_generate<subwarp_size, subwarps_per_block>(
        num_rows, m_row_ptrs, m_col_idxs, m_values, i_row_ptrs, i_col_idxs,
        i_values, excess_rhs_sizes, excess_nnz, trs_solve);
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void generate_excess_system(
    IndexType num_rows, const IndexType *__restrict__ m_row_ptrs,
    const IndexType *__restrict__ m_col_idxs,
    const ValueType *__restrict__ m_values,
    const IndexType *__restrict__ i_row_ptrs,
    const IndexType *__restrict__ i_col_idxs,
    const IndexType *__restrict__ excess_rhs_ptrs,
    const IndexType *__restrict__ excess_nz_ptrs,
    IndexType *__restrict__ excess_row_ptrs,
    IndexType *__restrict__ excess_col_idxs,
    ValueType *__restrict__ excess_values, ValueType *__restrict__ excess_rhs)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    if (row >= num_rows) {
        return;
    }

    const auto i_row_begin = i_row_ptrs[row];
    const auto i_row_size = i_row_ptrs[row + 1] - i_row_begin;

    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int local_id = subwarp.thread_rank();
    const auto prefix_mask = (config::lane_mask_type{1} << local_id) - 1;

    if (row == 0 && local_id == 0) {
        excess_row_ptrs[0] = 0;
    }

    if (i_row_size <= subwarp_size) {
        return;
    }

    auto excess_rhs_begin = excess_rhs_ptrs[row];
    auto excess_nz_begin = excess_nz_ptrs[row];

    // defer long rows: store their nnz and number of matches
    for (IndexType nz = 0; nz < i_row_size; ++nz) {
        auto col = i_col_idxs[i_row_begin + nz];
        auto m_row_begin = m_row_ptrs[col];
        auto m_row_size = m_row_ptrs[col + 1] - m_row_begin;
        // extract the sparse submatrix consisting of the entries whose
        // columns/rows match column indices from this row
        group_match<subwarp_size>(
            m_col_idxs + m_row_begin, m_row_size, i_col_idxs + i_row_begin,
            i_row_size, subwarp,
            [&](IndexType col, IndexType m_idx, IndexType i_idx,
                config::lane_mask_type mask, bool valid) {
                // trisystem(nz, i_idx) = m_values[m_row_begin + m_idx]
                // only in sparse :)
                if (valid) {
                    auto nz = excess_nz_begin + popcnt(mask & prefix_mask);
                    excess_col_idxs[nz] = excess_rhs_begin + i_idx;
                    excess_values[nz] = m_values[m_row_begin + m_idx];
                }
                excess_nz_begin += popcnt(mask);
            });
        if (local_id == 0) {
            // build right-hand side: 1 for diagonal entry, 0 else
            excess_rhs[excess_rhs_begin + nz] =
                row == col ? one<ValueType>() : zero<ValueType>();
            // store row pointers
            excess_row_ptrs[excess_rhs_begin + nz + 1] = excess_nz_begin;
        }
    }
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void copy_excess_solution(
    IndexType num_rows, const IndexType *__restrict__ i_row_ptrs,
    const IndexType *__restrict__ excess_rhs_ptrs,
    const ValueType *__restrict__ excess_solution,
    ValueType *__restrict__ i_values)
{
    const auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();

    if (row >= num_rows) {
        return;
    }

    auto local_id = threadIdx.x % subwarp_size;

    const auto i_row_begin = i_row_ptrs[row];

    const auto excess_begin = excess_rhs_ptrs[row];
    const auto excess_size = excess_rhs_ptrs[row + 1] - excess_begin;

    // if it was handled separately:
    if (excess_size > 0) {
        // copy the values for this row
        for (IndexType nz = local_id; nz < excess_size; nz += subwarp_size) {
            i_values[nz + i_row_begin] = excess_solution[nz + excess_begin];
        }
    }
}


}  // namespace kernel
