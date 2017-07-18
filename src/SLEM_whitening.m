function [Xwh, simi, lambda, alpha] = SLEM_whitening(Xposi, Xnega, lambda, alpha, mixed)

if ~exist('alpha', 'var')
    alpha = -.5;
end

if ~exist('mixed', 'var')
    mixed = 0;
end

if mixed
    X = [Xposi Xnega];
else
    X = Xnega;
end
t0 = tic;
[p, n] = size(X);
Nposi = size(Xposi, 2);

mu = mean(Xposi, 2);
A = cov(X')+lambda*eye(p);
[U,D] = svd(A);

%Sigma = U*(diag(diag(D).^alpha))*U';
Sigma = (diag(diag(D).^alpha))*U';
Xwh = Sigma*(Xposi-repmat(mu, 1, Nposi));

simi = Xwh'*Xwh;
NN = sqrt(diag(simi));
simi = simi./(NN*NN');
t1 = toc(t0)

%write_mc(['/scratch/sampaiod/online-e-svm/mc_files/whitening_alpha_' num2str(alpha) '_lambda_' num2str(lambda) '.mc'], Xposi, simi)