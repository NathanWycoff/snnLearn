// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// lif_forward_Cpp
List lif_forward_Cpp(int n_in, int l_h, IntegerVector n_h, int n_out, List Ws, List Fin, double t_eps, int t_steps);
RcppExport SEXP _snnLearn_lif_forward_Cpp(SEXP n_inSEXP, SEXP l_hSEXP, SEXP n_hSEXP, SEXP n_outSEXP, SEXP WsSEXP, SEXP FinSEXP, SEXP t_epsSEXP, SEXP t_stepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_in(n_inSEXP);
    Rcpp::traits::input_parameter< int >::type l_h(l_hSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type n_h(n_hSEXP);
    Rcpp::traits::input_parameter< int >::type n_out(n_outSEXP);
    Rcpp::traits::input_parameter< List >::type Ws(WsSEXP);
    Rcpp::traits::input_parameter< List >::type Fin(FinSEXP);
    Rcpp::traits::input_parameter< double >::type t_eps(t_epsSEXP);
    Rcpp::traits::input_parameter< int >::type t_steps(t_stepsSEXP);
    rcpp_result_gen = Rcpp::wrap(lif_forward_Cpp(n_in, l_h, n_h, n_out, Ws, Fin, t_eps, t_steps));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_snnLearn_lif_forward_Cpp", (DL_FUNC) &_snnLearn_lif_forward_Cpp, 8},
    {NULL, NULL, 0}
};

RcppExport void R_init_snnLearn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
