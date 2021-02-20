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

#ifndef GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL(_type)         \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::BatchDense<_type> *a,          \
                      const matrix::BatchDense<_type> *b,          \
                      matrix::BatchDense<_type> *c)

#define GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL(_type)         \
    void apply(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::BatchDense<_type> *alpha,      \
               const matrix::BatchDense<_type> *a,          \
               const matrix::BatchDense<_type> *b,          \
               const matrix::BatchDense<_type> *beta,       \
               matrix::BatchDense<_type> *c)

#define GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL(_type)         \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const matrix::BatchDense<_type> *alpha,      \
               matrix::BatchDense<_type> *x)

#define GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL(_type)         \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const matrix::BatchDense<_type> *alpha,      \
                    const matrix::BatchDense<_type> *x,          \
                    matrix::BatchDense<_type> *y)

#define GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL(_type)         \
    void add_scaled_diag(std::shared_ptr<const DefaultExecutor> exec, \
                         const matrix::BatchDense<_type> *alpha,      \
                         const matrix::Diagonal<_type> *x,            \
                         matrix::BatchDense<_type> *y)

#define GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL(_type)         \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     const matrix::BatchDense<_type> *x,          \
                     const matrix::BatchDense<_type> *y,          \
                     matrix::BatchDense<_type> *result)

#define GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL(_type)         \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec, \
                       const matrix::BatchDense<_type> *x,          \
                       matrix::BatchDense<remove_complex<_type>> *result)

#define GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL(_type)         \
    void count_nonzeros(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::BatchDense<_type> *source,     \
                        size_type *result)

#define GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(_type) \
    void calculate_max_nnz_per_row(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::BatchDense<_type> *source, size_type *result)

#define GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL(_type) \
    void calculate_nonzeros_per_row(                                     \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::BatchDense<_type> *source, Array<size_type> *result)

#define GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL(_type)         \
    void calculate_total_cols(std::shared_ptr<const DefaultExecutor> exec, \
                              const matrix::BatchDense<_type> *source,     \
                              size_type *result, size_type stride_factor,  \
                              size_type slice_size)

#define GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL(_type)         \
    void transpose(std::shared_ptr<const DefaultExecutor> exec, \
                   const matrix::BatchDense<_type> *orig,       \
                   matrix::BatchDense<_type> *trans)

#define GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL(_type)         \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::BatchDense<_type> *orig,       \
                        matrix::BatchDense<_type> *trans)

#define GKO_DECLARE_BATCH_DENSE_SYMM_PERMUTE_KERNEL(_vtype, _itype) \
    void symm_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                      const Array<_itype> *permutation_indices,     \
                      const matrix::BatchDense<_vtype> *orig,       \
                      matrix::BatchDense<_vtype> *permuted)

#define GKO_DECLARE_BATCH_DENSE_INV_SYMM_PERMUTE_KERNEL(_vtype, _itype) \
    void inv_symm_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                          const Array<_itype> *permutation_indices,     \
                          const matrix::BatchDense<_vtype> *orig,       \
                          matrix::BatchDense<_vtype> *permuted)

#define GKO_DECLARE_BATCH_DENSE_ROW_GATHER_KERNEL(_vtype, _itype) \
    void row_gather(std::shared_ptr<const DefaultExecutor> exec,  \
                    const Array<_itype> *gather_indices,          \
                    const matrix::BatchDense<_vtype> *orig,       \
                    matrix::BatchDense<_vtype> *row_gathered)

#define GKO_DECLARE_BATCH_DENSE_COLUMN_PERMUTE_KERNEL(_vtype, _itype) \
    void column_permute(std::shared_ptr<const DefaultExecutor> exec,  \
                        const Array<_itype> *permutation_indices,     \
                        const matrix::BatchDense<_vtype> *orig,       \
                        matrix::BatchDense<_vtype> *column_permuted)

#define GKO_DECLARE_BATCH_DENSE_INV_ROW_PERMUTE_KERNEL(_vtype, _itype)    \
    void inverse_row_permute(std::shared_ptr<const DefaultExecutor> exec, \
                             const Array<_itype> *permutation_indices,    \
                             const matrix::BatchDense<_vtype> *orig,      \
                             matrix::BatchDense<_vtype> *row_permuted)

#define GKO_DECLARE_BATCH_DENSE_INV_COLUMN_PERMUTE_KERNEL(_vtype, _itype)    \
    void inverse_column_permute(std::shared_ptr<const DefaultExecutor> exec, \
                                const Array<_itype> *permutation_indices,    \
                                const matrix::BatchDense<_vtype> *orig,      \
                                matrix::BatchDense<_vtype> *column_permuted)

#define GKO_DECLARE_BATCH_DENSE_EXTRACT_DIAGONAL_KERNEL(_vtype)        \
    void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec, \
                          const matrix::BatchDense<_vtype> *orig,      \
                          matrix::Diagonal<_vtype> *diag)

#define GKO_DECLARE_INPLACE_ABSOLUTE_BATCH_DENSE_KERNEL(_vtype) \
    void inplace_absolute_batch_dense(                          \
        std::shared_ptr<const DefaultExecutor> exec,            \
        matrix::BatchDense<_vtype> *source)

#define GKO_DECLARE_OUTPLACE_ABSOLUTE_BATCH_DENSE_KERNEL(_vtype) \
    void outplace_absolute_batch_dense(                          \
        std::shared_ptr<const DefaultExecutor> exec,             \
        const matrix::BatchDense<_vtype> *source,                \
        matrix::BatchDense<remove_complex<_vtype>> *result)

#define GKO_DECLARE_MAKE_COMPLEX_BATCH_DENSE_KERNEL(_vtype)        \
    void make_complex(std::shared_ptr<const DefaultExecutor> exec, \
                      const matrix::BatchDense<_vtype> *source,    \
                      matrix::BatchDense<to_complex<_vtype>> *result)

#define GKO_DECLARE_GET_REAL_BATCH_DENSE_KERNEL(_vtype)        \
    void get_real(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::BatchDense<_vtype> *source,    \
                  matrix::BatchDense<remove_complex<_vtype>> *result)

#define GKO_DECLARE_GET_IMAG_BATCH_DENSE_KERNEL(_vtype)        \
    void get_imag(std::shared_ptr<const DefaultExecutor> exec, \
                  const matrix::BatchDense<_vtype> *source,    \
                  matrix::BatchDense<remove_complex<_vtype>> *result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                         \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL(ValueType);                  \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL(ValueType);                         \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL(ValueType);                         \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL(ValueType);                    \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL(ValueType);               \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL(ValueType);                   \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL(ValueType);                 \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL(ValueType);                \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL(ValueType);     \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL(ValueType);    \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL(ValueType);          \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL(ValueType);                     \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL(ValueType);                \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_DENSE_SYMM_PERMUTE_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_DENSE_INV_SYMM_PERMUTE_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_DENSE_ROW_GATHER_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_DENSE_COLUMN_PERMUTE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_DENSE_INV_ROW_PERMUTE_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                        \
    GKO_DECLARE_BATCH_DENSE_INV_COLUMN_PERMUTE_KERNEL(ValueType, IndexType); \
    template <typename ValueType>                                            \
    GKO_DECLARE_BATCH_DENSE_EXTRACT_DIAGONAL_KERNEL(ValueType);              \
    template <typename ValueType>                                            \
    GKO_DECLARE_INPLACE_ABSOLUTE_BATCH_DENSE_KERNEL(ValueType);              \
    template <typename ValueType>                                            \
    GKO_DECLARE_OUTPLACE_ABSOLUTE_BATCH_DENSE_KERNEL(ValueType);             \
    template <typename ValueType>                                            \
    GKO_DECLARE_MAKE_COMPLEX_BATCH_DENSE_KERNEL(ValueType);                  \
    template <typename ValueType>                                            \
    GKO_DECLARE_GET_REAL_BATCH_DENSE_KERNEL(ValueType);                      \
    template <typename ValueType>                                            \
    GKO_DECLARE_GET_IMAG_BATCH_DENSE_KERNEL(ValueType)


namespace omp {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace omp


namespace cuda {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace cuda


namespace reference {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace reference


namespace hip {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace hip


namespace dpcpp {
namespace batch_dense {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_dense
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_
