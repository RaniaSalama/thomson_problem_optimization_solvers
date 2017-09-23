%
% testing the covergence behavior of Homotopy Method
%
Nit = 30;
X = [];
L = [];
%%% Homotopy converges much better with rho = 1.
rho = 10;
warning off;
disp(['Running Homotopy Method with matlab function ode45']);
disp(['Using ', num2str(Nit), ' different random starting points']);
for k = 1:Nit
   x0=rho*randn(3,1);
   [x, lambda] = Homotopy(@myfunc3,@myfunc3Jacob,x0);
   X    = [X, x];
   L    = [L length(lambda)];
end
