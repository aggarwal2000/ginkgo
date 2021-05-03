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

template <typename ValueType>
__global__ void generate_common_pattern_block_jacobi(
    const gko::batch_csr::UniformBatch<const ValueType> uni_mat_batch,
    int *const global_block_pointers_common_pattern,
    int *const global_pointers_to_blocks,
    int *const global_row_pointres_for_diagonal_blocks,
    int *const global_rows_inside_blocks, const int num_blocks_in_mat,
    const int blocks_arr_size)
{
    // TODO...
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < blocks_arr_size) {
        global_block_pointers_common_pattern[gid] = gid;
    }
}

template <typename ValueType>
class BatchBlockJacobi final {
public:
    /**
     * The size of the work vector required in case of static allocation.
     */
    static constexpr int shared_vals_work_size =
        batch_block_jacobi_config<ValueType>::max_blocks_arr_size;
    static constexpr int shared_idxs_work_size = 1;

    /**
     * The size of the work vector required in case of dynamic allocation.
     *
     */
    static constexpr int dynamic_shared_vals_work_size(int, int)
    {
        return batch_block_jacobi_config<ValueType>::max_blocks_arr_size;
    }

    static constexpr int dynamic_shared_idxs_work_size(int, int) { return 1; }

    __host__ __inline__ BatchBlockJacobi(
        std::shared_ptr<const CudaExecutor> exec,
        const gko::batch_csr::UniformBatch<const ValueType> &mat_batch,
        Array<int> &global_idxs_work_arr)
        : uni_mat_batch_{mat_batch}
    {
        // TODO: take in a vector of variable block sizes from the user
        std::vector<int> block_sizes_for_jacobi;
        const int side = mat_batch.num_rows;
        int total = 0;
        int curr_block_size;
        while (total < side) {
            curr_block_size = 4;
            int temp = total;
            temp += curr_block_size;
            if (temp > side) {
                curr_block_size =
                    side - total;  // In such  situation ,obviously, you won't
                                   // enter the loop in next iteration.
            }
            total = temp;
            block_sizes_for_jacobi.push_back(curr_block_size);
        }

        // block_sizes_for_jacobi is the vector containing block sizes for the
        // block jacobi preconditioner
        this->num_blocks_in_mat_ = block_sizes_for_jacobi.size();

        int blocks_arr_size_tmp = 0;
        for (int block_size : block_sizes_for_jacobi) {
            blocks_arr_size_tmp += block_size * block_size;
        }

        blocks_arr_size_ = blocks_arr_size_tmp;

        std::vector<int> blocks_ptrs_common_pattern(blocks_arr_size_, -1);
        std::vector<int> ptrs_to_blocks(this->num_blocks_in_mat_ + 1);
        std::vector<int> row_ptrs_for_diagonal_blocks(this->num_blocks_in_mat_ +
                                                      1);
        std::vector<int> rows_inside_blocks(
            uni_mat_batch_
                .num_rows);  // to keep track of which row is in which block

        ptrs_to_blocks[0] = 0;
        row_ptrs_for_diagonal_blocks[0] = 0;

        for (int i = 0; i < block_sizes_for_jacobi.size(); i++) {
            row_ptrs_for_diagonal_blocks[i + 1] =
                row_ptrs_for_diagonal_blocks[i] + block_sizes_for_jacobi[i];
            ptrs_to_blocks[i + 1] =
                ptrs_to_blocks[i] +
                block_sizes_for_jacobi[i] * block_sizes_for_jacobi[i];
        }


        int counter = 0;
        for (int i = 0; i < block_sizes_for_jacobi.size(); i++) {
            int sz = block_sizes_for_jacobi[i];

            while (sz > 0) {
                rows_inside_blocks[counter] = i;
                counter++;
                sz--;
            }
        }

        const int size_arr1 = blocks_arr_size_;
        const int size_arr2 = num_blocks_in_mat_ + 1;
        const int size_arr3 = num_blocks_in_mat_ + 1;
        const int size_arr4 = uni_mat_batch_.num_rows;

        int global_idxs_work_requirement =
            size_arr1 + size_arr2 + size_arr3 + size_arr4;
        global_idxs_work_arr.resize_and_reset(global_idxs_work_requirement);


        global_block_pointers_common_pattern_ = global_idxs_work_arr.get_data();
        global_pointers_to_blocks_ =
            global_block_pointers_common_pattern_ + size_arr1;
        global_row_pointres_for_diagonal_blocks_ =
            global_pointers_to_blocks_ + size_arr2;
        global_rows_inside_blocks_ =
            global_row_pointres_for_diagonal_blocks_ + size_arr3;

        exec->copy_from(gko::lend(exec->get_master()), size_arr1,
                        blocks_ptrs_common_pattern.data(),
                        global_block_pointers_common_pattern_);
        exec->copy_from(gko::lend(exec->get_master()), size_arr2,
                        ptrs_to_blocks.data(), global_pointers_to_blocks_);
        exec->copy_from(gko::lend(exec->get_master()), size_arr3,
                        row_ptrs_for_diagonal_blocks.data(),
                        global_row_pointres_for_diagonal_blocks_);
        exec->copy_from(gko::lend(exec->get_master()), size_arr4,
                        rows_inside_blocks.data(), global_rows_inside_blocks_);

        // TODO...
        dim3 grid(1);
        dim3 block(128);
        generate_common_pattern_block_jacobi<<<grid, block>>>(
            uni_mat_batch_, global_block_pointers_common_pattern_,
            global_pointers_to_blocks_,
            global_row_pointres_for_diagonal_blocks_,
            global_rows_inside_blocks_, num_blocks_in_mat_, blocks_arr_size_);
    }

    __device__ __forceinline__ void generate(
        const gko::batch_csr::BatchEntry<const ValueType> &mat,
        ValueType *const __restrict__ shared_work_vals,
        int *const __restrict__ shared_work_idxs)
    {
        matrix_entry_ = mat;
        shared_work_vals_entry_ = shared_work_vals;
        shared_work_idxs_entry_ = shared_work_idxs;

        // TODO...
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("\n\n %d ", global_block_pointers_common_pattern_[0]);
            printf("\n %d ", global_block_pointers_common_pattern_[1]);
            printf("\n %d ", global_block_pointers_common_pattern_[2]);
            printf("\n %d ", global_block_pointers_common_pattern_[3]);
            printf("\n %d ", global_block_pointers_common_pattern_[4]);
            printf("\n %d \n\n", global_block_pointers_common_pattern_[5]);
        }
    }

    __device__ __forceinline__ void apply(
        const gko::batch_dense::BatchEntry<const ValueType> &r,
        const gko::batch_dense::BatchEntry<ValueType> &z) const
    {
        // TODO
    }

private:
    // Note: It is mandatory to have these 4 data members in any preconditioner
    // implementation.
    ValueType *__restrict__ shared_work_vals_entry_ = nullptr;
    int *__restrict__ shared_work_idxs_entry_ = nullptr;
    gko::batch_csr::BatchEntry<const ValueType> matrix_entry_;
    const gko::batch_csr::UniformBatch<const ValueType> uni_mat_batch_;

    // Rest of the data members
    int num_blocks_in_mat_;
    int blocks_arr_size_;
    int *global_block_pointers_common_pattern_;
    int *global_pointers_to_blocks_;
    int *global_row_pointres_for_diagonal_blocks_;
    int *global_rows_inside_blocks_;
};