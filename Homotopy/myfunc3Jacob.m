function dy = myfunc3Jacob(xinput)
%
% test function for Newton Method
%

x = xinput(1);
y = xinput(2);
z = xinput(3);
m = xinput(4);
dy = [    2*x 2*y 0 0;
    0 0 2*z 2*m;
    (2*(2*x - 2*z)^2)/((m - y)^2 + (x - z)^2)^3 - 2/((m - y)^2 + (x - z)^2)^2 -(2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3 2/((m - y)^2 + (x - z)^2)^2 - (2*(2*x - 2*z)^2)/((m - y)^2 + (x - z)^2)^3 (2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3;
    -(2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3 (2*(2*m - 2*y)^2)/((m - y)^2 + (x - z)^2)^3 - 2/((m - y)^2 + (x - z)^2)^2 (2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3  2/((m - y)^2 + (x - z)^2)^2 - (2*(2*m - 2*y)^2)/((m - y)^2 + (x - z)^2)^3;
    2/((m - y)^2 + (x - z)^2)^2 - (2*(2*x - 2*z)^2)/((m - y)^2 + (x - z)^2)^3 (2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3 (2*(2*x - 2*z)^2)/((m - y)^2 + (x - z)^2)^3 - 2/((m - y)^2 + (x - z)^2)^2  -(2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3;
    (2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3 2/((m - y)^2 + (x - z)^2)^2 - (2*(2*m - 2*y)^2)/((m - y)^2 + (x - z)^2)^3 -(2*(2*m - 2*y)*(2*x - 2*z))/((m - y)^2 + (x - z)^2)^3 (2*(2*m - 2*y)^2)/((m - y)^2 + (x - z)^2)^3 - 2/((m - y)^2 + (x - z)^2)^2;

    ];