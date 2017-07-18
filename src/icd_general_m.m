function [G,pp,residual,time]=icd_general_m(K,tol,nmax)
t0 = tic;
n = size(K,1);
G = zeros(n,nmax);
pp = 1:n;
trK = trace(K);
diagK = diag(K);
residuals = diagK;
time = zeros(1,nmax);

residual=trK*ones(1,nmax);
[temp,jast] = max(residual);

for iter=1:nmax
    
    
    if (jast~=iter)
        % pivoting
        i=pp(jast);  pp(jast)=pp(iter);  pp(iter)=i;
        temp = G(jast,1:iter); G(jast,1:iter)=G(iter,1:iter); G(iter,1:iter)=temp;
    end
    G(iter,iter) =  sqrt(residuals(jast-iter+1));

    G(iter+1:n,iter) = K(pp(iter+1:n),pp(iter));
    if (iter>=2)
        G(iter+1:n,iter) = G(iter+1:n,iter) - G(iter+1:n,1:iter-1) * G(iter,1:iter-1)';
    end
            G(iter+1:n,iter) = G(iter+1:n,iter) / G(iter,iter);
    maxdiagG=0;
    residuals = diagK(pp(iter+1:n)) - sum(G(iter+1:n,1:iter).^2,2);
   % if any(residuals<0), keyboard; end
    residual(iter) = sum(residuals);
    [a,b] = max(residuals);
    jast = b + iter;
        if residual < tol * trK, break; end
    time(iter) = toc(t0);
end
G = G(:,1:iter);
residual = residual / trK;


