function [c,ceq] = constraints(x, k, n)
    x = reshape(x, [k, n]);
    ceq = 0;
    for i=1:n
        ceq = ceq + abs(norm(x(:,i))-1);
    end
    c = 0;
      