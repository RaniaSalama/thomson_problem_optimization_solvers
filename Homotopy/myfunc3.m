function eva = myfunc3(xinput)
%
% test function for Newton Method
%

x = xinput(1);
y = xinput(2);
z = xinput(3);
m = xinput(4);

eva = [
x^2 + y^2 - 1;
z^2 + m^2 - 1;
-2 * (x - z) * 1.0/((x-z)^2 + (y-m)^2)^2;
-2 * (y - m) * 1.0/((x-z)^2 + (y-m)^2)^2;
-2 * (z - x) * 1.0/((x-z)^2 + (y-m)^2)^2;
-2 * (m - y) * 1.0/((x-z)^2 + (y-m)^2)^2;

];

