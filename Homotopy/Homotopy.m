function [x, lambda] = Homotopy(Fun, dFun,x0)
% On input: 
%   Fun and dFun are the names of function and its Jacobian.
%   x0 is initial guess
%
% On output
%   flg = 0 means success; otherwise method failed.
%   x(:,end) is the solution if flg = 0.
%   
% Written by Ming Gu for Math 128B, Spring 2010
% 
[FunFcn,msg] = fcnchk(Fun,0);
if ~isempty(msg)
    error('InvalidFUN',msg);
end
[dFunFcn,msg] = fcnchk(dFun,0);
if ~isempty(msg)
    error('InvalidFUN',msg);
end
flg = 1;
x   = [];
x(:,1) = x0;
f0  = FunFcn(x0);
[Tout, Yout] = ode45(@(t,y)myfunc3ODE(t,y,f0,dFunFcn),[0,1],x0);
lambda = Tout(:);
x      = Yout(end,:)';

