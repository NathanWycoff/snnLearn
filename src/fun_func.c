/** src/fun_func.c Author "Nathan Wycoff <nathanbrwycoff@gmail.com>" Date 08.30.2018  */

double sumC(NumericVector x) {
  int n = x.size();
  double total = 0;
  for(int i = 0; i < n; ++i) {
    total += x[i];
  }
  return total;
}
