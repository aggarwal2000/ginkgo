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


// NOTE: Both the arrays must be sorted - in ascending order; starts:inclusive,
// ends: exclusive
template <int subwarp_size, typename Group, typename Callback>
__device__ __forceinline__ void group_match(
    const int* const __restrict__ arr1, const int arr1_st, const int arr1_end,
    const int* const __restrict__ arr2, const int arr2_st, const int arr2_end,
    Group subwarpgrp, Callback do_this_on_match)
{
    // TODO: replace by an efficient warp/subwarp parallel implementation
    // (something like- a parallel version of merge sort trick)
    const int id_in_subwarp_grp = subwarpgrp.thread_rank();

    for (int i = id_in_subwarp_grp + arr2_st; i < arr2_end; i += subwarp_size) {
        int arr2_val = arr2[i];

        for (int v = arr1_st; v < arr1_end; v++) {
            if (arr1[v] == arr2_val) {
                do_this_on_match(v, i);
                break;
            }
        }
    }
}


// NOTE: Both the arrays must be sorted - in ascending order; starts:inclusive,
// ends: exclusive
template <int subwarp_size, typename Group>
__device__ __forceinline__ void group_match(
    const int* const __restrict__ arr1, const int arr1_st, const int arr1_end,
    const int* const __restrict__ arr2, const int arr2_st, const int arr2_end,
    Group subwarpgrp, int& count_matches)
{
    // TODO: replace by an efficient warp/subwarp parallel implementation
    // (something like- a parallel version of merge sort trick)

    const int id_in_subwarp_grp = subwarpgrp.thread_rank();
    int count = 0;

    for (int i = id_in_subwarp_grp + arr2_st; i < arr2_end; i += subwarp_size) {
        int arr2_val = arr2[i];

        for (int v = arr1_st; v < arr1_end; v++) {
            if (arr1[v] == arr2_val) {
                count++;
                break;
            }
        }
    }

    subwarpgrp.sync();

    for (int offset = subwarp_size / 2; offset > 0; offset = offset / 2) {
        count += subwarpgrp.shfl_down(count, offset);
    }

    if (subwarpgrp.thread_rank() == 0) {
        count_matches = count;
    }
}

template <int subwarp_size>
__global__
    __launch_bounds__(default_block_size) void extract_dense_sys_pattern_kernel(
        const int nrows, const int* const __restrict__ A_row_ptrs,
        const int* const __restrict__ A_col_idxs, const int nnz_A,
        const int* const __restrict__ laiA_row_ptrs,
        const int* const __restrict__ laiA_col_idxs, const int nnz_laiA,
        int* const __restrict__ dense_mats_patterns,
        int* const __restrict__ rhs_one_idxs, int* const __restrict__ sizes,
        int* const __restrict__ count_matches_per_row_for_all_csr_sys)
{
    using gko::kernels::batch_isai::row_size_limit;
    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subgrpwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int id_within_warp = subwarpgrp.thread_rank();

    int ele_id_assigned_to_subwarpgrp = subgrpwarp_id_in_grid;

    for (int i_row_idx = 0; i_row_idx < nrows; i_row_idx++) {
        for (int nz_id = laiA_row_ptrs[i_row_idx];
             nz_id < laiA_row_ptrs[i_row_idx + 1]; nz_id++) {
            if (ele_id_assigned_to_subwarpgrp > nz_id) {
                continue;
            } else if (ele_id_assigned_to_subwarpgrp != nz_id) {
                continue;
            }

            // a subwarp grp deals with this ele in row: i_row_idx and col:
            // laiA_col_idxs[nz_id]
            int i_st = laiA_row_ptrs[i_row_idx];
            int i_end = laiA_row_ptrs[i_row_idx + 1];
            int i_size = i_end - i_st;
            int i_col_idx = laiA_col_idxs[nz_id];

            if (nz_id == i_st && subwarpgrp.thread_rank() == 0) {
                sizes[i_row_idx] = i_size;
            }

            if (i_col_idx == i_row_idx && subwarpgrp.thread_rank() == 0) {
                rhs_one_idxs[i_row_idx] = nz_id - i_st;
            }

            int m_row_idx = laiA_col_idxs[nz_id];
            // the subwarp_grp deals with this row in M
            int m_st = A_row_ptrs[m_row_idx];
            int m_end = A_row_ptrs[m_row_idx + 1];

            int& count_matches = count_matches_per_row_for_all_csr_sys[nz_id];

            if (i_size <= row_size_limit) {
                if (subwarpgrp.thread_rank() == 0) {
                    count_matches = -1;  // Not necessary
                }
                int* dense_ptr = dense_mats_patterns +
                                 row_size_limit * row_size_limit * i_row_idx;
                int* dense_ptr_for_row =
                    dense_ptr + (nz_id - i_st) * row_size_limit;

                auto do_this_on_match = [dense_ptr_for_row, i_st](
                                            int idx_match_arr1,
                                            int idx_match_arr2) {
                    dense_ptr_for_row[idx_match_arr1 - i_st] = idx_match_arr2;
                };
                group_match<subwarp_size>(laiA_col_idxs, i_st, i_end,
                                          A_col_idxs, m_st, m_end, subwarpgrp,
                                          do_this_on_match);

            } else {
                group_match<subwarp_size>(laiA_col_idxs, i_st, i_end,
                                          A_col_idxs, m_st, m_end, subwarpgrp,
                                          count_matches);
            }
            subwarpgrp.sync();
            ele_id_assigned_to_subwarpgrp += total_num_subwarp_grps_in_grid;
        }
    }
}


template <typename T, typename ValueType>
__device__ __forceinline__ ValueType solve_upper_tri_dense(
    T subwarp_grp, ValueType* const __restrict__ local_row, const int size)
{
    const int local_id = subwarp_grp.thread_rank();
    const ValueType rhs = local_id == size - 1 ? 1 : 0;
    ValueType sol = rhs;

    for (int dense_col_idx = size - 1; dense_col_idx >= 0; dense_col_idx--) {
        const ValueType ele = local_row[dense_col_idx];

        if (dense_col_idx == local_id) {
            sol = sol / ele;
        }

        subwarp_grp.sync();
        const ValueType bot = subwarp_grp.shfl(sol, dense_col_idx);

        if (local_id < dense_col_idx) {
            sol = sol - bot * ele;
        }
    }

    return sol;
}


template <typename T, typename ValueType>
__device__ __forceinline__ ValueType solve_lower_tri_dense(
    T subwarp_grp, ValueType* const __restrict__ local_row, const int size)
{
    const int local_id = subwarp_grp.thread_rank();
    const ValueType rhs = local_id == 0 ? 1 : 0;
    ValueType sol = rhs;

    for (int dense_col_idx = 0; dense_col_idx < size; dense_col_idx++) {
        const ValueType ele = local_row[dense_col_idx];

        if (dense_col_idx == local_id) {
            sol = sol / ele;
        }

        subwarp_grp.sync();
        const ValueType top = subwarp_grp.shfl(sol, dense_col_idx);

        if (local_id > dense_col_idx) {
            sol = sol - top * ele;
        }
    }

    return sol;
}


template <typename T, typename ValueType>
__device__ __forceinline__ int choose_pivot_row_1(
    T subwarpgrp, const int diag_pos, ValueType* const __restrict__ local_row,
    const int size)
{
    const int local_id = subwarpgrp.thread_rank();
    int piv_row_idx = local_id;
    ValueType val = local_row[diag_pos];
    const ValueType val1 = subwarpgrp.shfl(local_row[diag_pos], diag_pos);
    if (local_id >= size || local_id < diag_pos) {
        piv_row_idx = diag_pos;
        val = val1;
    }

    const int subwarp_size = subwarpgrp.size();

    for (int offset = subwarp_size / 2; offset > 0; offset = offset / 2) {
        subwarpgrp.sync();

        const ValueType val_other = subwarpgrp.shfl_down(val, offset);
        const int piv_row_idx_other = subwarpgrp.shfl_down(piv_row_idx, offset);
        if (abs(val_other) > abs(val)) {
            val = val_other;
            piv_row_idx = piv_row_idx_other;
        }
    }

    // 0th thread has correct piv_row_idx
    subwarpgrp.sync();
    piv_row_idx = subwarpgrp.shfl(piv_row_idx, 0);
    return piv_row_idx;
}


template <typename T, typename ValueType>
__device__ __forceinline__ void swap_rows_and_rhs(
    T subwarpgrp, const int diag_pos, const int pivot_row_idx,
    ValueType* const __restrict__ local_row, const int size, ValueType& rhs)
{
    const int local_id = subwarpgrp.thread_rank();

    for (int col = 0; col < size; col++) {
        ValueType diag_tid_col_val = subwarpgrp.shfl(local_row[col], diag_pos);
        ValueType piv_row_tid_col_val =
            subwarpgrp.shfl(local_row[col], pivot_row_idx);

        if (local_id == diag_pos) {
            local_row[col] = piv_row_tid_col_val;
        }

        if (local_id == pivot_row_idx) {
            local_row[col] = diag_tid_col_val;
        }
    }

    ValueType diag_tid_rhs = subwarpgrp.shfl(rhs, diag_pos);
    ValueType piv_row_tid_rhs = subwarpgrp.shfl(rhs, pivot_row_idx);

    if (local_id == diag_pos) {
        rhs = piv_row_tid_rhs;
    }

    if (local_id == pivot_row_idx) {
        rhs = diag_tid_rhs;
    }
}


template <typename T, typename ValueType>
__device__ __forceinline__ void row_transformation(
    T subwarpgrp, const int diag_pos, ValueType* const __restrict__ local_row,
    const int size, ValueType& rhs)
{
    const int local_id = subwarpgrp.thread_rank();
    const ValueType diag_ele = subwarpgrp.shfl(local_row[diag_pos], diag_pos);
    if (diag_ele == 0) {
        printf(
            "\n Got 0 at diag position while solving general dense linear "
            "system...\n");
        // assert(0);
    }
    const ValueType multiplier = local_row[diag_pos] / diag_ele;
    const ValueType rhs_key_val = subwarpgrp.shfl(rhs, diag_pos);

    for (int col = 0; col < size; col++) {
        const ValueType col_key_val = subwarpgrp.shfl(local_row[col], diag_pos);
        if (local_id != diag_pos) {
            local_row[col] -= multiplier * col_key_val;
        }
    }

    if (local_id != diag_pos) {
        rhs -= multiplier * rhs_key_val;
    }
}


template <typename T, typename ValueType>
__device__ __forceinline__ ValueType
solve_general_dense(T subwarpgrp, ValueType* const __restrict__ local_row,
                    const int size, const int rhs_one_idx)
{
    const int local_id = subwarpgrp.thread_rank();
    ValueType rhs = rhs_one_idx == local_id ? 1 : 0;

    for (int diag_pos = 0; diag_pos < size; diag_pos++) {
        const int pivot_row_idx =
            choose_pivot_row_1(subwarpgrp, diag_pos, local_row, size);
        if (pivot_row_idx != diag_pos) {
            swap_rows_and_rhs(subwarpgrp, diag_pos, pivot_row_idx, local_row,
                              size, rhs);
        }

        subwarpgrp.sync();
        row_transformation(subwarpgrp, diag_pos, local_row, size, rhs);
        subwarpgrp.sync();
    }

    rhs = rhs / local_row[local_id];
    return rhs;
}


template <int subwarp_size, typename ValueType>
__global__
    __launch_bounds__(default_block_size) void fill_values_dense_mat_and_solve_kernel(
        const size_type nbatch, const int nrows,
        const int* const __restrict__ laiA_row_ptrs,
        const int* const __restrict__ laiA_col_idxs,
        ValueType* const __restrict__ laiA_values, const int nnz_laiA,
        const int* const __restrict__ A_row_ptrs,
        const int* const __restrict__ A_col_idxs,
        const ValueType* const __restrict__ A_values, const int nnz_A,
        const int* const dense_patterns,
        const int* const __restrict__ rhs_one_idxs,
        const int* const __restrict__ sizes, const int matrix_type)
{
    using gko::kernels::batch_isai::row_size_limit;
    static_assert(row_size_limit <= subwarp_size, "incompatible subwarp size");

    auto subwarpgrp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    const int subwarp_id_in_grid =
        thread::get_subwarp_id_flat<subwarp_size, int>();
    const int total_num_subwarp_grps_in_grid =
        thread::get_subwarp_num_flat<subwarp_size, int>();
    const int local_id = subwarpgrp.thread_rank();

    for (int i = subwarp_id_in_grid; i < nrows * nbatch;
         i += total_num_subwarp_grps_in_grid) {
        const int row = i % nrows;
        const int batch_id = i / nrows;

        const int* dense_ptr =
            dense_patterns + row_size_limit * row_size_limit * row;
        const int size = sizes[row];
        const int rhs_one_idx = rhs_one_idxs[row];

        if (size > row_size_limit) {
            continue;
        }

        ValueType local_row[row_size_limit];
        // Note: The Pattern shouldn't be  a transpose.

        // implicit transpose
        for (int k = 0; k < size; k++)  // colasced access by subwarp_grp
        {
            const int val_idx = dense_ptr[k * row_size_limit + local_id];

            if (val_idx != -1) {
                local_row[k] = A_values[val_idx + batch_id * nnz_A];
            } else {
                local_row[k] = 0;
            }
        }

        subwarpgrp.sync();

        ValueType lai_A_ele;
        // Now solve the system
        if (matrix_type == 0)  // i.e. lower
        {
            lai_A_ele = solve_upper_tri_dense(subwarpgrp, local_row, size);
        } else if (matrix_type == 1)  // i.e. upper
        {
            lai_A_ele = solve_lower_tri_dense(subwarpgrp, local_row, size);
        } else if (matrix_type == 2)  // i.e. dense
        {
            lai_A_ele =
                solve_general_dense(subwarpgrp, local_row, size, rhs_one_idx);
        } else {
            printf("\n No such case: line: %d and file: %s ", __LINE__,
                   __FILE__);
            assert(false);
        }

        if (local_id < size) {
            laiA_values[batch_id * nnz_laiA + laiA_row_ptrs[row] + local_id] =
                lai_A_ele;  // colasced access by subwarp_grp
        }
    }
}
