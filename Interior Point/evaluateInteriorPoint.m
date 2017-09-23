function [time_array, f_x, xs, xs_norm] = evaluateInteriorPoint(n, k, random_iterations_number)
    time_array = zeros(random_iterations_number,1);
    f_x = zeros(random_iterations_number,1);
    xs_norm = zeros(random_iterations_number,1);
    xs = zeros(k*n,1);
    for i=1:random_iterations_number
        xs = rand(k* n,1)';
        tic;
        thomposon_function = @(x) func(x, k, n);
        thompson_constraints = @(x) constraints(x, k, n);
        options = optimoptions('fmincon');
        options.MaxFunctionEvaluations = 20000;
        options.MaxIterations = 1000;
        options.ConstraintTolerance = 1.0000e-12;

        x = fmincon(thomposon_function,xs,[],[],[],[],[],[],thompson_constraints,options);
        time = toc;
        time_array(i) = time;
        f_x(i) = func(x, k, n);
        xshaped = reshape(x, [k, n]);
        xs_norm(i) = sum(abs(computeNorms(xshaped, n) - ones(n,1)));
        xs = x;
    end