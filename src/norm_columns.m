function X = norm_columns(X)
for i = 1:size(X, 2)
    X(:,i) = X(:,i)/norm(X(:,i));
end
