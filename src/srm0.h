/** src/srm0.h Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 11.21.2018  */

#ifndef SRM0_H
#define SRM0_H

double **par_sim_body_c(int *net_shape, const int n_layers,
        double **Fin, int *f_count_in, long long int **f_max, double ***Ws,
        int** f_count, const int t_steps, const double t_eps, double ****GAMMA, double ****GAMMAd, const int debug, const bool copy_gamma);

#endif
