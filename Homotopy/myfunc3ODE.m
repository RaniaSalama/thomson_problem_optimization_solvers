function y = myfunc3ODE(t,xinput, f0,dfun)
%
% Homotopy function used in ODE solver.
%
dy = feval(dfun,xinput);

y  = - dy \f0;
