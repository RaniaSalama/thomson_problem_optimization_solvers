function norms=computeNorms(x, n)
    norms = zeros(n,1);
    for i=1:n
        norms(i) = norm(x(:,i));
    end