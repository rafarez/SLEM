function [Wposi, simi] = new_full_encoder(Xposi, Xnega, kernel_type, k, theta, lambda, omega_normalized, varargin)

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
        
        Delta = Wposi-repmat(mu, 1, Nposi);% \in p\times Nposi
        C = 2./(theta*sum(Delta.*(invA*Delta))+theta+1);% \in 1\times Nposi
        % U = A + theta/(theta+1)*(x_0-mu)*(x_0-mu)'
    
        % W_kk = zeros(p,Nposi);
        beta_kk = repmat(C, p, 1).*(invA*Delta);
        % b_kk = zeros(1,Nposi);
        nu_kk = (theta-1)/(theta+1) -1/(theta+1)*diag(beta_kk'*(theta*Wposi+repmat(mu,1,Nposi)))';
        
        Wposi = [nu_kk; beta_kk];
        if omega_normalized
            for i=1:Nposi
                Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
            end
        end
        clear W_kk
        
        if kk<k
            mean_scores = mean(diag(Wposi(:,1:100)'*[ones(1, 100); Xposi(:,1:100)]));
            Delta_n = Wnega-repmat(mu, 1, n);% \in p\times n
            theta_n = theta+((1+mean_scores)^2)/((1-mean_scores)^2)/2;% + 4/n/
            C_n = 1./(theta_n*sum(Delta_n.*(invA*Delta_n))+theta_n+1);% \in 1\times n
            
            beta_kk = repmat(C_n, p, 1).*(invA*Delta_n);
            % b_kk = zeros(1,Nposi);
            nu_kk = (theta_n-1)/(theta_n+1) -1/(theta_n+1)*diag(beta_kk'*(theta_n*Wnega+repmat(mu,1,n)))';
        
            Wnega = [nu_kk; beta_kk];
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
        [B, perm] = incomplete_chol(Wnega, kernel_type, rank_max, prec);
        
        r = size(B, 2);
        I = perm(1:r);
        mu = mean(B)';
        S = 1/n*(B'*B);
        %invA = preprocessing_regression_parameters(S, lambda);
        A = S -mu*mu' +lambda*eye(r);
        invA = A\eye(r);
        %clear G S
        
        t1 = toc(t0);
        disp(['pre-processing negative data time: ' num2str(t1)])
        
        %% processing Wposi
        t2 = tic;
        %Phi = (kernel_matrix(Wposi', Wnega(:,I), kernel_type)*G)';% r\times Nposi
        %k_00 = diag(kernel_matrix(Wposi', Wposi, kernel_type))';% 1\times Nposi
        %% line to be changed for k>1
        k_0 = kernel_matrix(Wnega(:,I)', Xposi, kernel_type); % r\times Nposi
            
        b_0 = B(1:r,:)\k_0; % r\times Nposi
        %b_00 = (B\(B'*B))*k_0;
        delta = b_00-repmat(mu, 1, Nposi);
        C = 2./(1 +1/theta +diag(delta'*(invA*delta))');
        
        beta_kk = repmat(C, r, 1).*(invA*delta);
        nu_kk = (theta-1)/(theta+1)-1/(theta+1)*diag( beta_kk'*(theta*b_00+repmat(mu, 1, Nposi)) )';
        
        Wposi = [nu_kk; beta_kk];
        if omega_normalized
            for i=1:Nposi
                Wposi(:,i) = Wposi(:,i)/norm(Wposi(:,i));
            end
        end
        clear beta_kk

        t3 = toc(t2);
        disp(['processing positive data time: ' num2str(t3)])
        
        %% processing Wnega
        if kk<k
            t4 = tic;
            invA = [1/lambda, zeros(1,r); zeros(r,1), invG];
            v = B';% r\times n 
            u = zeros(1, n);
            Delta_n = [u; v-repmat(mu, 1, n)];% (r+1)\times n
            C = 2./(1 +1/theta +diag(Delta_n'*(invA*Delta_n))');% 1\times Nposi
            
            beta_kk = repmat(C, r+1, 1).*(invA*Delta_n);
            nu_kk = (theta-1)/(theta+1)-1/(theta+1)*diag(beta_kk'*([theta*u; theta*v+repmat(mu, 1, n)]))';
            
            Wnega = [nu_kk; beta_kk];
            if omega_normalized
                for i=1:Nposi
                    Wnega(:,i) = Wnega(:,i)/norm(Wnega(:,i));
                end
            end
            clear W_kk
            t5 = toc(t4);
            disp(['processing positive data time: ' num2str(t5)])
        end
        
    end
end

simi = Wposi'*Wposi;