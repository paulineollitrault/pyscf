# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cc SHARED
  ccsd_pack.c ccsd_grad.c ccsd_t.c ccsd_t_lambda.c uccsd_t.c
  dlpno_foo_dressed.c dlpno_per_i.c dlpno_partner.c
  dlpno_b_tilde.c dlpno_g_tilde.c dlpno_t1_ints.c
  dlpno_c_tilde.c dlpno_d_tilde.c dlpno_t1_fock.c
  dlpno_t1_residual.c dlpno_be.c dlpno_g_term.c
  dlpno_cd_term.c dlpno_t34.c dlpno_be_v2.c
  dlpno_pair_centerQ.c dlpno_cross_partner.c
  dlpno_partners_centerQ.c dlpno_s_pno.c
  dlpno_u_for_triple.c dlpno_triple_local_df.c
  dlpno_triple_tno.c dlpno_triples_full.c
  dlpno_w3_full.c dlpno_triples_orch.c
  dlpno_g_tilde_plan.c
  dlpno_g_tilde_enum.c
  dlpno_s_pno_all.c
  dlpno_lmp2_residual.c
  dlpno_ccsd_solver.cpp)
add_dependencies(cc np_helper ao2mo)

set_target_properties(cc PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

target_link_libraries(cc ao2mo cvhf np_helper ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})

