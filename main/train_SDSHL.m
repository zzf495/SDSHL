function [B] = train_SDSHL(XTrain,YTrain,LTrain,param)
%% Official codes of SDSHL
%%%     Authors                      Teng et al.
%%%
%%%     Title                        Joint Specifics and Dual-Semantic Hashing 
%%%                                     Learning for Cross-Modal Retrieval
%%%
%% Intput
%%%
%%%     XTrain                       The feature of the first modality
%%%
%%%     YTrain                       The feature of the second modality
%%%
%%%     LTrain                       The label set of samples
%%%
%% param
%%%
%%%     nbits                        The length of hash codes
%%%
%%%     max_iter                     The number of iteration
%%%
%%%     k                            The number of clusters/anchors
%%%
%%%     alpha                        The weight of manifold term
%%%
%%%     eta1/eta2                    The weight of A^t A^t' in Eq. (6)
%%%
%%%     beta                         The weight of manifold \| A^t - A^p \|
%%%
%%%     gamma                        The weight of regularization w.r.t. A^t,
%%%                                  W^t
%%%
%%%     theta                        The weight of \|G^t-R^t W^t L \|
%%%
%%%     lambda1                      The weight of \| rS - B' G^t \|
%%%
%%%     lambda2                      The weight of \| rS - G^t' G^p \|
%%%
%%%     xi                           The weight of regularization  of Eq. (27)
%%%
%% Output
%%%
%%%     B                            Learned hashing code of samples
%%%
%% Version
%%%
%%%     Upload                   2024-04-03
%%%
    %% parameter setting
    max_iter = param.max_iter;
    alpha = param.alpha; 
    beta = param.beta;
    lambda1 = param.lambda1;
    lambda2 = param.lambda2;
    gamma = param.gamma;
    theta = param.theta;
    eta1 = param.eta1;
    eta2 = param.eta2;
    nbits = param.nbits;
    k = param.k;
    K = k-1;
    n = size(LTrain,2);
    c = size(LTrain,1); 
    L = LTrain;

    %% init
    [index1,C1_init] = kmeans(XTrain',k);
    [index2,C2_init] = kmeans(YTrain',k);

    R1 = randn(nbits,nbits);
    [U11, ~, ~] = svd(R1);
    R1 = U11(:,1:nbits);
    clear U11;

    R2 = randn(nbits,nbits);
    [U11, ~, ~] = svd(R2);
    R2 = U11(:,1:nbits);
    clear U11;

    W1 = randn(nbits,c);
    W2 = randn(nbits,c);

    A1 = randn(n,k);
    A2 = randn(n,k);

    B = sign(randn(nbits,n));
    V1 = randn(nbits,n);
    V2 = randn(nbits,n);

    a = (c*(c+2)+c*sqrt(c*(c+2)))/4+eps;

    %% iter
    norm_loss = cell(1,7);
    for i=1:7
        norm_loss{1,i}=[];
    end
    clear i;
  
    for iter = 1:max_iter 
        disp("iter = "+iter);
        
        % U-Step:
        ZU1 = XTrain*LTrain'*W1';
        U1 = myOrth(ZU1);
        
        ZU2 = YTrain*LTrain'*W2';
        U2 = myOrth(ZU2);


        if(iter == 1)
            C1 = U1'*C1_init';
            C2 = U2'*C2_init';
        end
        % W-Step:
        W1 = (U1'*XTrain*LTrain'+theta*R1'*V1*LTrain'+alpha*C1*A1'*LTrain')...
            /(gamma*eye(c) + alpha*LTrain*A1*A1'*LTrain' + (1+theta)*LTrain*LTrain');
        W2 = (U2'*YTrain*LTrain'+theta*R2'*V2*LTrain'+alpha*C2*A2'*LTrain')...
            /(gamma*eye(c) + alpha*LTrain*A2*A2'*LTrain' + (1+theta)*LTrain*LTrain');

        % C-Step:
        for j=1:k
            % update C1
            temp1 = W1*LTrain;
            sub_idx=find(index1==j);
            up = temp1(:,sub_idx)*A1(sub_idx,j);
            down = sum(A1(sub_idx,j));
            if(down==0)
                C1(:,j) = zeros(length(up),1);
            elseif(down~=0)
                C1(:,j) = up/down;
            end
            clear up down sub_idx;
            
            % update C2
            temp2 = W2*LTrain;
            sub_idx=find(index2==j);
            up = temp2(:,sub_idx)*A2(sub_idx,j);
            down = sum(A2(sub_idx,j));
            if(down==0)
                C2(:,j) = zeros(length(up),1);
            elseif(down~=0)
                C2(:,j) = up/down;
            end
        end

        % A-Step:
        % d(||x_i - C_j||^2_2)
        % A1
        ed1 = L2_distance_1(W1*LTrain, C1);
        [~, idxx1] = sort(ed1,2); % sort each row
        for i = 1:n
            id = idxx1(i,1:K+1);
            di = ed1(i,id); 
            numerator = alpha*di(K+1)-alpha*di+2*beta*A2(i,id(:))-2*beta*A2(i,id(K+1));
            denominator1 = K*alpha*di(K+1)-alpha*sum(di(1:K));
            denominator2 = 2*beta*sum(A2(i,id(1:K)))-2*K*beta*A2(i,id(K+1));
            A1(i,id) = max(numerator/(denominator1+denominator2+eps),0);
        end
        % A2
        ed2 = L2_distance_1(W2*LTrain, C2);
        [~, idxx2] = sort(ed2,2); % sort each row
        for i = 1:n
            id = idxx2(i,1:K+1);
            di = ed2(i,id);
            numerator = alpha*di(K+1)-alpha*di+2*beta*A1(i,id(:))-2*beta*A1(i,id(K+1));
            denominator1 = K*alpha*di(K+1)-alpha*sum(di(1:K));
            denominator2 = 2*beta*sum(A1(i,id(1:K)))-2*K*beta*A1(i,id(K+1));
            A2(i,id) = max(numerator/(denominator1+denominator2+eps),0);
        end

         % R-Step:
        [left1,~,right1] = svd(V1*LTrain'*W1');
        R1 = left1*right1';
        [left2,~,right2] = svd(V2*LTrain'*W2');
        R2 = left2*right2';


        % V-Step:
        % V1
        ZV1 = lambda1*nbits/(a*c+1)*(a*B*L'*L+eta1*B*A1*A1'+eta2*B*A2*A2')...
            +lambda2*nbits/(a*c+1)*(a*V2*L'*L+eta1*V2*A1*A1'+eta2*V2*A2*A2')...
            +theta*R1*W1*LTrain;
     
        Temp = ZV1*ZV1'-1/n*(ZV1*ones(n,1)*(ones(1,n)*ZV1'));
        [~,Lmd,QQ] = svd(Temp);
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (ZV1'-1/n*ones(n,1)*(ones(1,n)*ZV1')) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V1 = sqrt(n)*[Q Q_]*[P P_]';
        clear Q P Q_ P_ Lmd QQ Temp;
        
        % V2
        ZV2 = lambda1*nbits/(a*c+1)*(a*B*L'*L+eta1*B*A1*A1'+eta2*B*A2*A2')...
            +lambda2*nbits/(a*c+1)*(a*V1*L'*L+eta1*V1*A1*A1'+eta2*V1*A2*A2')...
            +theta*R2*W2*LTrain;
        Temp = ZV2*ZV2'-1/n*(ZV2*ones(n,1)*(ones(1,n)*ZV2'));
        [~,Lmd,QQ] = svd(Temp);
        idx = (diag(Lmd)>1e-6);
        Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
        P = (ZV2'-1/n*ones(n,1)*(ones(1,n)*ZV2')) *  (Q / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        V2 = sqrt(n)*[Q Q_]*[P P_]';
        clear Q P Q_ P_ Lmd QQ Temp;

        % B-Step:
        t1 = a*V1*L'*L+eta1*V1*A1*A1'+eta2*V1*A2*A2';
        t2 = a*V2*L'*L+eta1*V2*A1*A1'+eta2*V2*A2*A2';
        B = sign(nbits/(a*c+1)*(t1+t2));

        X = U1'*XTrain;Y = U2'*YTrain;
        [index1,~] = kmeans(X',k,'Start',C1');
        [index2,~] = kmeans(Y',k,'Start',C2');

    end
    B(B==0) = -1;
end