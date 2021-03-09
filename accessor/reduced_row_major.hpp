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

#ifndef GKO_ACCESSOR_REDUCED_ROW_MAJOR_HPP_
#define GKO_ACCESSOR_REDUCED_ROW_MAJOR_HPP_


#include <array>
#include <memory>
#include <type_traits>
#include <utility>


#include "accessor_helper.hpp"
#include "accessor_references.hpp"
#include "index_span.hpp"
#include "range.hpp"
#include "utils.hpp"


namespace gko {
/**
 * @brief The accessor namespace.
 *
 * @ingroup accessor
 */
namespace acc {


/**
 * The reduced_row_major class allows a storage format that is different from
 * the arithmetic format (which is returned from the brace operator).
 * As storage, the StorageType is used.
 *
 * This accessor uses row-major access. For example for three dimensions,
 * neighboring z coordinates are next to each other in memory, followed by y
 * coordinates and then x coordinates.
 *
 * @tparam Dimensionality  The number of dimensions managed by this accessor
 *
 * @tparam ArithmeticType  Value type used for arithmetic operations and
 *                         for in- and output
 *
 * @tparam StorageType  Value type used for storing the actual value to memory
 *
 * @note  This class only manages the accesses and not the memory itself.
 */
template <int Dimensionality, typename ArithmeticType, typename StorageType>
class reduced_row_major {
public:
    using arithmetic_type = std::remove_cv_t<ArithmeticType>;
    using storage_type = StorageType;
    static constexpr size_type dimensionality{Dimensionality};
    static constexpr bool is_const{std::is_const<storage_type>::value};
    using const_accessor =
        reduced_row_major<dimensionality, arithmetic_type, const storage_type>;

    static_assert(Dimensionality >= 1,
                  "Dimensionality must be a positive number!");

    friend class range<reduced_row_major>;

protected:
    using storage_stride_type = std::array<size_type, dimensionality - 1>;
    using reference_type =
        reference_class::reduced_storage<arithmetic_type, storage_type>;

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param size  multidimensional size of the memory
     * @param storage  pointer to the block of memory containing the storage
     * @param stride  stride array used for memory accesses
     */
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major(
        std::array<size_type, dimensionality> size, storage_type *storage,
        storage_stride_type stride)
        : size_(size), storage_{storage}, stride_(stride)
    {}

    /**
     * Creates the accessor for an already allocated storage space with a
     * stride. The first stride is used for computing the index for the first
     * index, the second stride for the second index, and so on.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     * @param strides  strides used for memory accesses
     */
    template <typename... Strides>
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major(
        std::array<size_type, dimensionality> size, storage_type *storage,
        Strides &&... strides)
        : reduced_row_major{
              size, storage,
              storage_stride_type{{std::forward<Strides>(strides)...}}}
    {
        static_assert(sizeof...(Strides) + 1 == dimensionality,
                      "Number of provided Strides must be dimensionality - 1!");
    }

    /**
     * Creates the accessor for an already allocated storage space.
     * It is assumed that all accesses are without padding.
     *
     * @param storage  pointer to the block of memory containing the storage
     * @param size  multidimensional size of the memory
     */
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major(
        std::array<size_type, dimensionality> size, storage_type *storage)
        : reduced_row_major{
              size, storage,
              helper::compute_default_row_major_stride_array<size_type>(size)}
    {}

    /**
     * Creates an empty accessor (pointing nowhere with an empty size)
     */
    constexpr GKO_ACC_ATTRIBUTES reduced_row_major()
        : reduced_row_major{{0, 0, 0}, nullptr}
    {}

public:
    /**
     * Creates a reduced_row_major range which contains a read-only version of
     * the current accessor.
     *
     * @returns  a reduced_row_major major range which is read-only.
     */
    constexpr GKO_ACC_ATTRIBUTES range<const_accessor> to_const() const
    {
        return range<const_accessor>{size_, storage_, stride_};
    }

    /**
     * Returns the length in dimension `dimension`.
     *
     * @param dimension  a dimension index
     *
     * @returns  length in dimension `dimension`
     */
    constexpr GKO_ACC_ATTRIBUTES size_type length(size_type dimension) const
    {
        return dimension < dimensionality ? size_[dimension] : 1;
    }

    /**
     * Copies data from another accessor
     *
     * @warning Do not use this function since it is not optimized for a
     *          specific executor. It will always be performed sequentially.
     *          Please write an optimized version (adjusted to the architecture)
     *          by iterating through the values yourself.
     *
     * @tparam OtherAccessor  type of the other accessor
     *
     * @param other  other accessor
     */
    template <typename OtherAccessor>
    GKO_ACC_ATTRIBUTES void copy_from(const OtherAccessor &other) const
    {
        helper::multidim_for_each(size_, [this, &other](auto... indices) {
            (*this)(indices...) = other(indices...);
        });
    }

    /**
     * Returns the stored value for the given indices. If the storage is const,
     * a value is returned, otherwise, a reference is returned.
     *
     * @param indices  indices which value is supposed to access
     *
     * @returns  the stored value if the accessor is const (if the storage type
     *           is const), or a reference if the accessor is non-const
     */
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES std::enable_if_t<
        are_all_integral<Indices...>::value,
        std::conditional_t<is_const, arithmetic_type, reference_type>>
    operator()(Indices &&... indices) const
    {
        return reference_type{storage_ +
                              compute_index(std::forward<Indices>(indices)...)};
    }

    /**
     * Returns a sub-range spanning the current range (x1_span, x2_span, ...)
     *
     * @param spans  span for the indices
     *
     * @returns a sub-range for the given spans.
     */
    template <typename... SpanTypes>
    constexpr GKO_ACC_ATTRIBUTES
        std::enable_if_t<helper::are_index_span_compatible<SpanTypes...>::value,
                         range<reduced_row_major>>
        operator()(SpanTypes... spans) const
    {
        return helper::validate_index_spans(size_, spans...),
               range<reduced_row_major>{
                   std::array<size_type, dimensionality>{
                       (index_span{spans}.end - index_span{spans}.begin)...},
                   storage_ + compute_index((index_span{spans}.begin)...),
                   stride_};
    }

    /**
     * Returns the size of the accessor
     *
     * @returns the size of the accessor
     */
    constexpr GKO_ACC_ATTRIBUTES std::array<size_type, dimensionality>
    get_size() const
    {
        return size_;
    }

    /**
     * Returns a pointer to a stride array of size dimensionality - 1
     *
     * @returns returns a pointer to a stride array of size dimensionality - 1
     */
    GKO_ACC_ATTRIBUTES
    constexpr const storage_stride_type &get_stride() const { return stride_; }

    /**
     * Returns the pointer to the storage data
     *
     * @returns the pointer to the storage data
     */
    constexpr GKO_ACC_ATTRIBUTES storage_type *get_stored_data() const
    {
        return storage_;
    }

    /**
     * Returns a const pointer to the storage data
     *
     * @returns a const pointer to the storage data
     */
    constexpr GKO_ACC_ATTRIBUTES const storage_type *get_const_storage() const
    {
        return storage_;
    }

protected:
    template <typename... Indices>
    constexpr GKO_ACC_ATTRIBUTES size_type
    compute_index(Indices &&... indices) const
    {
        static_assert(sizeof...(Indices) == dimensionality,
                      "Number of indices must match dimensionality!");
        return helper::compute_row_major_index<size_type, dimensionality>(
            size_, stride_, std::forward<Indices>(indices)...);
    }

private:
    const std::array<size_type, dimensionality> size_;
    storage_type *storage_;
    const storage_stride_type stride_;
};


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_REDUCED_ROW_MAJOR_HPP_
