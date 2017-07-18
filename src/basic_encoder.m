function [Wposi, simi, para_theta, para_lambda] = basic_encoder(Xposi, Xnega, theta, lambda, w_normalized)
%% rewritting naive form, to compare with quick_basic_encoder
%disp(X)
disp('RELS-SVM: Recursive Exemplar Least-Squares SVM');
disp('Non-kernelized, no bias');
disp('----------------------------------------');

[p, n] = size(Xnega);
Nposi = size(Xposi, 2);

mu = mean(Xnega,2);
%S = 1/n*(Xnega*Xnega');
A = 1/n*(Xnega*Xnega')+lambda*eye(p);
%invA = A\eye(p);
    
Wposi = zeros(p,Nposi);
for i=1:Nposi
    t0=tic;
    x_0 = Xposi(:,i);
    U = A + theta*(x_0*x_0');

    %% using Woodbury identity
    %invU = invA - theta/(theta*x_0'*invA*x_0+1)*(invA*x_0*x_0'*invA);
    w = U\(theta*x_0-mu);
    %w = invU*(theta*x_0-mu);
    if w_normalized
        Wposi(:,i) = w/norm(w);
    else
        Wposi(:,i) = w;
    end
    t1 = toc(t0);
    disp(t1)
end
       
simi = Wposi'*Wposi;

%% for cross validation in parallel
para_theta = theta;
para_lambda = lambda;