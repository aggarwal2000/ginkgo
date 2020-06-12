/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/solver/multigrid.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/solver/multigrid_kernels.hpp"

namespace gko {
namespace solver {
namespace multigrid {


GKO_REGISTER_OPERATION(initialize, multigrid::initialize);


}  // namespace multigrid


namespace {


template <typename ValueType>
void handle_list(
    std::shared_ptr<const Executor> &exec, size_type index,
    std::shared_ptr<const LinOp> &matrix,
    std::vector<std::shared_ptr<const LinOpFactory>> &smoother_list,
    gko::Array<ValueType> &relaxation_array,
    std::vector<std::shared_ptr<LinOp>> &smoother,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &relaxation,
    std::shared_ptr<matrix::Dense<ValueType>> &one)
{
    auto list_size = smoother_list.size();
    if (list_size != 0) {
        auto temp_index = list_size == 1 ? 0 : index;
        GKO_ENSURE_IN_BOUNDS(temp_index, list_size);
        auto item = smoother_list.at(temp_index);
        if (item == nullptr) {
            smoother.emplace_back(nullptr);
        } else {
            smoother.emplace_back(give(item->generate(matrix)));
        }
    } else {
        smoother.emplace_back(nullptr);
    }
    auto array_size = relaxation_array.get_num_elems();
    if (array_size != 0) {
        auto temp_index = array_size == 1 ? 0 : index;
        GKO_ENSURE_IN_BOUNDS(temp_index, array_size);
        auto data = relaxation_array.get_data();
        relaxation.emplace_back(give(matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{1},
            Array<ValueType>::view(exec, 1, data + temp_index), 1)));
    } else {
        // default is one;
        relaxation.emplace_back(one);
    }
}


}  // namespace


template <typename ValueType>
void Multigrid<ValueType>::generate()
{
    // generate coarse matrix until reaching max_level or min_coarse_rows
    auto num_rows = system_matrix_->get_size()[0];
    size_type level = 0;
    auto matrix = system_matrix_;
    auto exec = this->get_executor();
    // Always generate smoother and relaxation with size = level.
    while (level < parameters_.max_levels &&
           num_rows > parameters_.min_coarse_rows) {
        auto index = rstr_prlg_index_(level, lend(matrix));
        GKO_ENSURE_IN_BOUNDS(index, parameters_.rstr_prlg.size());
        auto rstr_prlg_factory = parameters_.rstr_prlg.at(index);
        // pre_smooth_generate
        handle_list(exec, index, matrix, parameters_.pre_smoother,
                    parameters_.pre_relaxation, pre_smoother_list_,
                    pre_relaxation_list_, one_op_);
        // mid_smooth_generate
        if (parameters_.mid_case == multigrid_mid_uses::mid) {
            handle_list(exec, index, matrix, parameters_.mid_smoother,
                        parameters_.mid_relaxation, mid_smoother_list_,
                        mid_relaxation_list_, one_op_);
        }
        // post_smooth_generate
        if (!parameters_.post_uses_pre) {
            handle_list(exec, index, matrix, parameters_.post_smoother,
                        parameters_.post_relaxation, post_smoother_list_,
                        post_relaxation_list_, one_op_);
        }
        // coarse generate
        auto rstr = rstr_prlg_factory->generate(matrix);
        rstr_prlg_list_.emplace_back(give(rstr));
        matrix = rstr_prlg_list_.back()->get_coarse_operator();
        num_rows = matrix->get_size()[0];
        level++;
    }
    if (parameters_.post_uses_pre) {
        post_smoother_list_ = pre_smoother_list_;
        post_relaxation_list_ = pre_relaxation_list_;
    }
    if (parameters_.mid_case == multigrid_mid_uses::pre) {
        mid_smoother_list_ = pre_smoother_list_;
        mid_relaxation_list_ = pre_relaxation_list_;
    } else if (parameters_.mid_case == multigrid_mid_uses::post) {
        mid_smoother_list_ = post_smoother_list_;
        mid_relaxation_list_ = post_relaxation_list_;
    }
    // generate coarsest solver
    if (parameters_.coarsest_solver != nullptr) {
        coarsest_solver_ = parameters_.coarsest_solver->generate(matrix);
    } else {
        // default is identity
        coarsest_solver_ =
            matrix::Identity<ValueType>::create(exec, matrix->get_size()[0]);
    }
}


template <typename ValueType>
void Multigrid<ValueType>::prepare_vcycle(
    const size_type nrhs,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e) const
{
    auto current_nrows = system_matrix_->get_size()[0];
    auto exec = this->get_executor();
    for (int i = 0; i < rstr_prlg_list_.size(); i++) {
        auto next_nrows =
            rstr_prlg_list_.at(i)->get_coarse_operator()->get_size()[0];
        r.at(i) = matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{current_nrows, nrhs});
        g.at(i) = matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{next_nrows, nrhs});
        e.at(i) = matrix::Dense<ValueType>::create(
            exec, gko::dim<2>{next_nrows, nrhs});
        current_nrows = next_nrows;
    }
}


template <typename ValueType>
void Multigrid<ValueType>::run_cycle(
    multigrid_cycle cycle, size_type level, std::shared_ptr<const LinOp> matrix,
    const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *x,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &r_list,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &g_list,
    std::vector<std::shared_ptr<matrix::Dense<ValueType>>> &e_list) const
{
    auto r = r_list.at(level);
    auto g = g_list.at(level);
    auto e = e_list.at(level);
    // get the pre_smoother
    auto pre_smoother = pre_smoother_list_.at(level);
    auto pre_relaxation = pre_relaxation_list_.at(level);
    // get the post_smoother
    auto post_smoother = post_smoother_list_.at(level);
    auto post_relaxation = post_relaxation_list_.at(level);
    // get the mid_smoother
    auto mid_smoother = mid_smoother_list_.at(level);
    auto mid_relaxation = mid_relaxation_list_.at(level);
    // move the residual computation in level zero to out-of-cycle
    if (level != 0) {
        r->copy_from(b);
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
    // x += relaxation * Smoother(r)
    if (pre_smoother) {
        pre_smoother->apply(pre_relaxation.get(), r.get(), one_op_.get(), x);
        // compute residual
        r->copy_from(b);  // n * b
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
    // first cycle
    rstr_prlg_list_.at(level)->restrict_apply(r.get(), g.get());
    // next level or solve it
    if (level + 1 == rstr_prlg_list_.size()) {
        coarsest_solver_->apply(g.get(), e.get());
    } else {
        this->run_cycle(cycle, level + 1,
                        rstr_prlg_list_.at(level)->get_coarse_operator(),
                        g.get(), e.get(), r_list, g_list, e_list);
    }
    // additional work for non-v_cycle
    if (cycle == multigrid_cycle::f || cycle == multigrid_cycle::w) {
        // second cycle - f_cycle, w_cycle
        // prolong
        rstr_prlg_list_.at(level)->prolong_applyadd(e.get(), x);
        // compute residual
        r->copy_from(b);  // n * b
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        // mid-smooth
        if (mid_smoother) {
            mid_smoother->apply(mid_relaxation.get(), r.get(), one_op_.get(),
                                x);
            // compute residual
            r->copy_from(b);  // n * b
            matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        }

        rstr_prlg_list_.at(level)->restrict_apply(r.get(), g.get());
        // next level or solve it
        if (level + 1 == rstr_prlg_list_.size()) {
            coarsest_solver_->apply(g.get(), e.get());
        } else {
            if (cycle == multigrid_cycle::f) {
                // f_cycle call v_cycle in the second cycle
                this->run_cycle(
                    multigrid_cycle::v, level + 1,
                    rstr_prlg_list_.at(level)->get_coarse_operator(), g.get(),
                    e.get(), r_list, g_list, e_list);
            } else {
                this->run_cycle(
                    cycle, level + 1,
                    rstr_prlg_list_.at(level)->get_coarse_operator(), g.get(),
                    e.get(), r_list, g_list, e_list);
            }
        }
    } else if (cycle == multigrid_cycle::kfcg ||
               cycle == multigrid_cycle::kgcr) {
        // do some work in coarse level - do not need prolong
        GKO_NOT_IMPLEMENTED;
    }

    // prolong
    rstr_prlg_list_.at(level)->prolong_applyadd(e.get(), x);

    // post-smooth
    if (post_smoother) {
        r->copy_from(b);
        matrix->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
        post_smoother->apply(post_relaxation.get(), r.get(), one_op_.get(), x);
    }
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    if (cycle_ == multigrid_cycle::kfcg || cycle_ == multigrid_cycle::kgcr) {
        GKO_NOT_IMPLEMENTED;
    }

    auto exec = this->get_executor();
    constexpr uint8 RelativeStoppingId{1};
    Array<stopping_status> stop_status(exec, b->get_size()[1]);
    bool one_changed{};
    auto dense_x = gko::as<matrix::Dense<ValueType>>(x);
    auto dense_b = gko::as<matrix::Dense<ValueType>>(b);
    std::vector<std::shared_ptr<vector_type>> r_list(rstr_prlg_list_.size());
    std::vector<std::shared_ptr<vector_type>> g_list(rstr_prlg_list_.size());
    std::vector<std::shared_ptr<vector_type>> e_list(rstr_prlg_list_.size());
    this->prepare_vcycle(b->get_size()[1], r_list, g_list, e_list);
    exec->run(multigrid::make_initialize(e_list, &stop_status));
    // compute the residual at the r_list(0);
    auto r = r_list.at(0);
    r->copy_from(dense_b);
    system_matrix_->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, r.get());
    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(this, iter, r.get(),
                                                            dense_x);
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }
        this->run_cycle(cycle_, 0, system_matrix_, dense_b, dense_x, r_list,
                        g_list, e_list);
        r->copy_from(dense_b);
        system_matrix_->apply(neg_one_op_.get(), x, one_op_.get(), r.get());
    }
}


template <typename ValueType>
void Multigrid<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                      const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_MULTIGRID(_type) class Multigrid<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID);


}  // namespace solver
}  // namespace gko
