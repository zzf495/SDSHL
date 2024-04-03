function output = myOrth(input)

    [input_m, input_n] = size(input);
    tmp = input' * input -1 / input_m * (input' * ones(input_m,1) *(ones(1,input_m) * input));
    [~, SIGMA, RIGHT] = svd(tmp); clear tmp
    IDX = (diag(SIGMA) > 1e-6);
    RIGHT1 = RIGHT(:, IDX); RIGHT2 = orth(RIGHT(:, ~IDX));
    LEFT1 = (input - 1 / input_m * ones(input_m,1) * (ones(1,input_m) * input)) *  (RIGHT1 / (sqrt(SIGMA(IDX, IDX))));
    LEFT2 = orth(randn(input_m, input_n - length(find(IDX == 1))));
    [a, b] = size(LEFT2);
    if a < input_m || b < input_n - length(find(IDX == 1))
        output = (sqrt(input_m) * LEFT1 * RIGHT1');
    else
        output = (sqrt(input_m) * [LEFT1 LEFT2] * [RIGHT1 RIGHT2]');
    end
end