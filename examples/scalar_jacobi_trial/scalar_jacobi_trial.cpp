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

// @sect3{Include files}

// This is the main ginkgo header file.
#include <ginkgo/ginkgo.hpp>

// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <iostream>
// Add the STL map header for the executor selection
#include <map>
// Add the string manipulation header to handle strings.
#include <string>


int main(int argc, char *argv[])
{
    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor] " << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                 true);
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};

    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    auto A = gko::share(
        gko::read<gko::matrix::Csr<>>(std::ifstream("data/A.mtx"), exec));
    auto b = gko::read<gko::matrix::Dense<>>(std::ifstream("data/b.mtx"), exec);
    auto x =
        gko::read<gko::matrix::Dense<>>(std::ifstream("data/x0.mtx"), exec);
    auto x_clone = gko::clone(x);

    auto precond_factory_object =
        gko::preconditioner::Scalarjacobi<>::build().on(exec);

    auto precond_object = precond_factory_object->generate(gko::share(A));

    auto inv_ele_vals = precond_object->get_inv_eles();

    auto precond_object_cpu =
        gko::clone(gko::ReferenceExecutor::create(), precond_object);

    std::cout << "\nPrining the inv elements array of scalar jacobi "
                 "preconditioner\n\n";
    for (size_t i = 0; i < precond_object_cpu->get_num_stored_elements(); i++) {
        std::cout << " i : " << i << "  -->  "
                  << precond_object_cpu->get_inv_eles()[i] << std::endl;
    }

    std::cout << "\n\nApply the preconditioner\n\n";
    precond_object->apply(lend(b), lend(x_clone));

    std::cout << "\n\n The preconditioned vector is: \n\n";
    write(std::cout, lend(x_clone));


    std::cout << "\n\n#########################################################"
                 "###############################################"
              << std::endl
              << std::endl;

    std::cout << "\n Create solver factory\n\n";

    auto solver_factory =
        gko::solver::Cg<>::build()
            .with_generated_preconditioner(gko::share(precond_object))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(20u).on(exec),
                gko::stop::ResidualNorm<>::build()
                    .with_reduction_factor(1e-15)
                    .on(exec))
            .on(exec);

    std::cout << "\n\n Create solver object\n\n";

    auto solver = solver_factory->generate(gko::share(A));

    std::cout << "\n\n Now solve system\n\n";

    solver->apply(lend(b), lend(x));

    std::cout << "\n\n The solution is:\n\n ";

    write(std::cout, lend(x));
}
