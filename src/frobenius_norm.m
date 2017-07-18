function out = frobenius_norm(Xnega, B, perm, kernel_type)

[n, r] = size(B);

K = kernel_matrix(Xnega(:,perm)', Xnega(:,perm), kernel_type);
normK = norm(K, 'fro');
disp(normK)
step = 50;
idx = [0, mod(r, step):step:r];
out = zeros(length(idx)-1, 1);
for i = 1:length(out)
    K = K -B(:, idx(i)+1:idx(i+1))*B(:, idx(i)+1:idx(i+1))';
    out(i) = norm(K, 'fro')/normK;
    disp(out(i))
end

figure; 
plot(idx(2:end), out, 'b');