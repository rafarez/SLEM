function [Wposi, simi] = quick_full_encoder(Xposi, Xnega, kernel_type, k, theta, lambda, omega_normalized, varargin)


[p, n] = size(Xnega);
Nposi = size(Xposi, 2);
%w_null = 0;

if isempty(varargin)
    rank_max = n;
    prec = 1e-8;
elseif length(varargin) == 1
    rank_max = varargin{1};
    prec = 1e-8;
else
    rank_max = varargin{1};
    prec = varargin{2};
end

if isequal(kernel_type{1}, 'None')
    Wposi = Xposi;
    Wnega = Xnega;
    %disp(X)
    disp('RSLEM: Recursive Square Loss Exemplar Machine');
    disp('Non-kernelized');
    disp('----------------------------------------');
    for kk=1:k
        t0 = tic;
        p = size(Wposi, 1);
        % kk-th interation
        disp(['Iteration ' num2str(kk) '/' num2str(k)]);
        
        mu = mean(Wnega,2);
        % A = 1/n*(Wnega*Wnega') - mu*mu' + lambda*eye(p);
        invA = (1/n*(Wnega*Wnega') - mu*mu' + lambda*eye(p))\eye(p);
        
        delta = Wposi-repmat(mu, 1, Nposi);% \in p\times Nposi
        C = 2./(sum(delta.*(invA*delta))+1/theta+1);% \in 1\times Nposi
        % U = A + theta/(theta+1)*(x_0-mu)*(x_0-mu)'
    
        % W_kk = zeros(p,Nposi);
        beta_kk = repmat(C, p, 1).*(invA*delta);
        % b_kk = zeros(1,Nposi);
        nu_kk = (theta-1)/(theta+1) -1/(theta+1)*diag(beta_kk'*(theta*Wposi+repmat(mu,1,Nposi)))';
        
        Wposi = beta_kk;%[nu_kk; beta_kk];
        if omega_normalized
            for i=1:Nposi
                Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
            end
        end
        clear W_kk
        
        if kk<k
            %mean_scores = mean(diag(Wposi(:,1:100)'*[ones(1, 100); Xposi(:,1:100)]));
            Delta_n = Wnega-repmat(mu, 1, n);% \in p\times n
            theta_n = theta;%+((1+mean_scores)^2)/((1-mean_scores)^2)/2;% + 4/n/
            C_n = 2./(sum(Delta_n.*(invA*Delta_n))+1/theta_n+1);% \in 1\times n
            
            beta_kk = repmat(C_n, p, 1).*(invA*Delta_n);
            % b_kk = zeros(1,Nposi);
            nu_kk = (theta_n-1)/(theta_n+1) -1/(theta_n+1)*diag(beta_kk'*(theta_n*Wnega+repmat(mu,1,n)))';
        
            Wnega = beta_kk;%[nu_kk; beta_kk];
            if omega_normalized
                for i=1:n
                    Wnega(:,i) = Wnega(:,i)/norm(Wnega(:,i));
                end
            end
            clear W_kk
        end
        t1 = toc(t0);
        disp(['Iteration time : ' num2str(t1)])
    end
    simi = Wposi'*Wposi;
else
    Wposi = Xposi;
    Wnega = Xnega;
    %disp(X)
    disp('RSLEM: Recursive Square Loss Exemplar Machine');
    disp(['Kernel: ' kernel_type{1}]);
    disp('----------------------------------------');
    for kk=1:k
        % kk-th interation
        disp(['Iteration ' num2str(kk) '/' num2str(k)]);
        %% pre-processing Wnega
        t0 = tic;
        [B, perm, mu, r, w_null] = preprocessing_negative_data(Wnega, kernel_type, rank_max, prec);
        %invA = preprocessing_regression_parameters(S, lambda);
        %G = S -mu*mu' +lambda*eye(r);
        invG = (1/n*(B'*B) -mu*mu' +lambda*eye(r))\eye(r);
        %clear G S
        
        %% let's cheat...
        w_null = 1;
        t1 = toc(t0);
        disp(['pre-processing negative data time: ' num2str(t1)])
        
        %% processing Wposi
        t2 = tic;
        k_00 = kernel_matrix(Wposi', Wposi, kernel_type);% Nposi \times Nposi
        k_0 = kernel_matrix(Wnega(:,perm)', Wposi, kernel_type); %n\times Nposi
            
        if ~w_null
            %disp('w? wtf?')
            v = B(1:r,:)\k_0(1:r,:); % r\times Nposi
            u = sqrt(diag(k_00)' - diag(v'*v)');% 1\times Nposi
            %u = zeros(1, Nposi);
            w = zeros(n, Nposi);
            w(r+1:n,:) = (1./repmat(u, n-r, 1)).*(k_0(r+1:n,:) - B(r+1:n,:)*v);
            wbar = mean(w);% 1\times Nposi
            %Delta = [u-wbar; v-repmat(mu,1,Nposi)];% r+1\times Nposi
        
            a_00 = 1/n*diag(w'*w)' - wbar.^2 + lambda;% 1\times Nposi
            a_0  = 1/n*B'*w - mu*wbar;% r\times Nposi
            gamma = 1./(a_00 - diag(a_0'*(invG*a_0))');% 1\times Nposi
            xi = u -wbar -diag(a_0'*invG*(v-repmat(mu,1,Nposi)))';% 1\times Nposi
        
            C = 2./(1 +1/theta + (u-wbar).*gamma.*xi -gamma.*xi.*(diag((v-repmat(mu, 1, Nposi))'*invG*a_0)') +diag( (v-repmat(mu, 1, Nposi))'*invG*(v-repmat(mu, 1, Nposi)) )' );% 1\times Nposi
        
            beta_kk = repmat(C, r+1, 1).*[gamma.*xi; -repmat(gamma.*xi, r, 1).*(invG*a_0)+invG*(v-repmat(mu, 1, Nposi)) ];
            nu_kk = (theta-1)/(theta+1) -1/(theta+1)*diag(beta_kk'*([theta*u+wbar; theta*v+repmat(mu,1,Nposi)]))';
    
            Wposi = [nu_kk; beta_kk];
            if omega_normalized
                for i=1:Nposi
                    Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
                end
            end
            clear W_kk
            
            if kk<k
                Wnega = [];
            end
        
        else
            %disp('No w, hurray!')
            %invA = [1/lambda, zeros(1,r); zeros(r,1), invG]; clear invG
            %Bplus = (B'*B)\B';% r\times n
            v = ((B'*B)\B')*k_0;% r\times Nposi
            u = sqrt(diag(k_00)'-diag(v'*v)');% 1\times Nposi
            
            delta = [u; v-repmat(mu, 1, Nposi)];% (r+1)\times Nposi
            %C = 2./(1 +1/theta +1/lambda*(u.*u) +diag((v-repmat(mu, 1, Nposi))'*invG*(v-repmat(mu, 1, Nposi)))');
            C = 2./(1 +1/theta +diag(delta'*([1/lambda, zeros(1,r); zeros(r,1), invG]*delta))');% 1\times Nposi
            
            %beta_kk = repmat(C, r+1, 1).*(invA*delta);
            
            beta_0   = C.*u/lambda;%1\times Nposi
            beta_hat = repmat(C, r, 1).*(invG*(v-repmat(mu, 1, Nposi)));%r\times Nposi
            %beta_kk = [beta_0; beta_hat];
            %nu_kk = (theta-1)/(theta+1)-1/(theta+1)*diag(beta_kk'*([theta*u; theta*v+repmat(mu, 1, Nposi)]))';
            
            %P = B/(B'*B);% n\times r
            %alpha_0 = beta_0./u;% 1\times Nposi
            %alpha_hat = repmat(-alpha_0, n, 1).*(P*v) + P*beta_hat;% n\times Nposi
            
            if omega_normalized
                for i = 1:Nposi
                    M = norm([beta_0(i); beta_hat(:,i)]);
                    beta_0(i) = beta_0(i)/M;
                    beta_hat(:,i) = beta_hat(:,i)/M;
                end
            end
            Wposi = [beta_0; beta_hat];
        end
        t3 = toc(t2);
        disp(['processing positive data time: ' num2str(t3)])
        
        %% processing Wnega
        if kk<k
            t4 = tic;
            %invA = [1/lambda, zeros(1,r); zeros(r,1), invG];
            v = B';% r\times n 
            u = zeros(1, n);
            Delta_n = [u; v-repmat(mu, 1, n)];% (r+1)\times n
            C = 2./(1 +1/theta +diag(Delta_n'*([1/lambda, zeros(1,r); zeros(r,1), invG]*Delta_n))');% 1\times Nposi
            
            beta_0   = C.*u/lambda;
            beta_hat = repmat(C, r+1, 1).*(invG*(v-repmat(mu, 1, Nposi)));
            %nu_kk = (theta-1)/(theta+1)-1/(theta+1)*diag(beta_kk'*([theta*u; theta*v+repmat(mu, 1, n)]))';
            
            alpha_0 = zeros(1, n);
            alpha_hat = repmat(-alpha_0, n, 1).*(P*v) + P*beta_hat;
            Wnega = beta_kk;%[nu_kk; beta_kk];
            if omega_normalized
                for i = 1:Nposi
                    M = norm([alpha_0(i); alpha_hat(:,i)]);
                    alpha_0(i) = alpha_0(i)/M;
                    alpha_hat(:,i) = alpha_hat(:,i)/M;
                end
            end
            Wposi = [alpha_0; alpha_hat];
            t5 = toc(t4);
            disp(['processing positive data time: ' num2str(t5)])
        end
        
    end
    %% alpha 
%     P = B/(B'*B);% n\times r
%     alpha_0 = C/lambda;% 1\times Nposi
%     alpha_hat = repmat(-alpha_0, n, 1).*(P*v) + P*beta_hat;% n\times Nposi
%     if omega_normalized
%         for i = 1:Nposi
%             M = norm([alpha_0(i); alpha_hat(:,i)]);
%             alpha_0(i) = alpha_0(i)/M;
%             alpha_hat(:,i) = alpha_hat(:,i)/M;
%         end
%     end
%     simi = beta_hat'*beta_hat + ( (beta_0'*beta_0)./(u'*u) ).*(k_00c - v'*v);
%     norme = sqrt(diag(simi));
%     simi_beta  = simi./(norme*norme');
%     simi_alpha = alpha_hat'*(B*B')*alpha_hat + (alpha_0'*alpha_0).*k_00c ...
%         + repmat(alpha_0', 1, Nposi).*(k_0'*alpha_hat) + repmat(alpha_0, Nposi, 1).*(alpha_hat'*k_0);
    %% similarity calculation
    simi = (invG*(v-repmat(mu, 1, Nposi)))'*(invG*(v-repmat(mu, 1, Nposi))) + (k_00-v'*v)/lambda/lambda;
end

