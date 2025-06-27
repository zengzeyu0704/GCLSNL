function [Y]= GCLSNL(X,Y_tensorT,Omega,Par)
% X - Observed data
% Y_tensorT - Original data
% Omega - Observation index
% Par - Parameter set
Nway = size(X);
Maxiter = 50;

%% 
mu      = Par.alpha./Par.tau;
alpha   = Par.alpha;
beta1   = Par.beta1;
beta2   = Par.beta2;
beta3   = Par.beta3;
beta4   = Par.beta4;
lambda1 = Par.lambda1;
lambda2 = Par.lambda2;
lambda3 = Par.lambda3;
lambda4 = Par.lambda4;

%% 
J1 = zeros(Nway);
J2 = J1;
J3 = J1;
W1 = J1;
W2 = J1;
W3 = J1;
W4 = J1;
tol = 1e-3;

%% 
D1 = zeros(Nway(1),Nway(1),Nway(3));
a = diag(-ones(Nway(1),1));
b = diag(ones(Nway(1)-1,1),1);
tempD = a+b;
tempD(end,1) = 1;
D1(:,:,1) = tempD;

D2=zeros(Nway(2),Nway(2),Nway(3));
a=diag(-ones(Nway(2),1));
b=diag(ones(Nway(2)-1,1),-1);
tempD=a+b;
tempD(1,end)=1;
D2(:,:,1) = tempD;

D3=zeros(Nway(1),Nway(1),Nway(3));
D3(:,:,1) =  -eye(Nway(1));
D3(:,:,end) = eye(Nway(1));

I = zeros(Nway(1),Nway(1),Nway(3));
I(:,:,1)=eye(Nway(1));

Fa = (1/sqrt(Nway(1)))*fft(eye(Nway(1)));
Fb = (1/sqrt(Nway(2)))*fft(eye(Nway(2)));

for k = 1 : Maxiter
    oldY = X;
    %% update M
    X1 = permute(X,[2,3,1]);  X2 = permute(X,[3,1,2]);  X3 = X;
    j1 = permute(J1,[2,3,1]); j2 = permute(J2,[3,1,2]); j3 = J3;
    
    tau = alpha./mu;
    M1 = ipermute(prox_tnn_my(X1+j1/mu(1),tau(1)),[2,3,1]);
    M2 = ipermute(prox_tnn_my(X2+j2/mu(2),tau(2)),[3,1,2]);
    M3 = prox_tnn_my(X3+j3/mu(3),tau(3));   
    
    %% update L
    temp1 = lambda1./beta1;
    temp2 = lambda2./beta2;
    temp3 = lambda3./beta3;
    L1 = prox_l1(tprod(D1,X)+W1/beta1,temp1);
    L2 = prox_l1(tprod(X,D2)+W2/beta2,temp2);
    L3 = prox_l1(tprod(D3,X)+W3/beta3,temp3);
    
    %% update N
    temp = X+W4/beta4;
    parfor i=1:Nway(3)
        [~,t] = BM3D(1,temp(:,:,i),sqrt(lambda4/beta4)); 
        N(:,:,i)=t;
    end
    
    %% update X
    tempA = (sum(mu)+beta4)*I+beta1*tprod(tran(D1),D1)+beta3*tprod(tran(D3),D3);
    tempB = beta2*tprod(D2,tran(D2));
    temp1 = mu(1)*M1-J1+mu(2)*M2-J2+mu(3)*M3-J3+beta2*tprod(L2-W2/beta2,tran(D2));
    temp2 = beta1*tprod(tran(D1),L1-W1/beta1);
    temp3 = beta3*tprod(tran(D3),L3-W3/beta3);
    temp4 = beta4*(N-W4/beta4);
    tempC = temp1+temp2+temp3+temp4;
    
    tempAf=fft(tempA,[],3);
    tempBf=fft(tempB,[],3);
    tempCf=fft(tempC,[],3);
    
    % compute X
    Xf = zeros(Nway);
    for i=1:Nway(3)
        Ai=tempAf(:,:,i);
        Bi=tempBf(:,:,i);
        Ci=tempCf(:,:,i);
        da=Ai(:,1); deigA=fft(da);
        db=Bi(:,1); deigB=fft(db);  
        Sig=repmat(deigA,1,Nway(2))+repmat(deigB',Nway(1),1);
        Sig=1./Sig;
        temp=Sig.*(Fa*Ci*Fb');
        Xf(:,:,i)=Fa'*temp*Fb;
    end

    X = real(ifft(Xf,[],3));
    
    Y = X;
    X(Omega) = Y_tensorT(Omega);
    Y(Omega) = Y_tensorT(Omega);
    
    %% 
    res=norm(Y(:)-oldY(:))/norm(oldY(:));
    
    if res < tol
        break;
    end
    
    %% 
    J1 = J1+mu(1)*(X-M1);
    J2 = J2+mu(2)*(X-M2);
    J3 = J1+mu(3)*(X-M3);
    W1 = W1+beta1*(tprod(D1,X)-L1);
    W2 = W2+beta2*(tprod(X,D2)-L2);
    W3 = W3+beta3*(tprod(D3,X)-L3);
    W4 = W4+beta4*(X-N);
    
end

end