function fx = func(x, k, n)
    x = reshape(x, [k, n]);
    fx = 0;
    for i=1:n
        for j=1:i
            if i ~= j
                fx = fx + (1.0/norm(x(:,i) - x(:,j))^2);
            end
        end
    end