clc;clear;
% PCA用NIPALS方法实现
% [Xnormal,~,Xbad] = data_sample(4000,10,0.3,0.4,0.5);
load data

X_mean = mean(Xnormal);
X_std = std(Xnormal);
Xnormal  = Xnormal - ones(size(Xnormal,1),1)*X_mean;   % 正常数据去平均值
Xnormal  = Xnormal/diag(X_std);            % 归一化
Xbad = Xbad-ones(length(Xbad),1)*X_mean;
Xbad = Xbad/diag(X_std);

X = Xnormal;
np = 3;
[nX,mX] = size(X);
T = zeros(nX,np);
P = zeros(mX,np);
option = [1e-8 2000];
% [T,P,ssq,Ro,Rv,Lo,Lv] = mypca(Xnormal,3,[1e-8 2000 0]);
for a = 1:np
    iter = 0;
    [aa,aaa] = max(sum(X.^2,1));
    T(:,a) = X(:,aaa);
    t_old = T(:,a)*10;
    while (sum((t_old - T(:,a)).^2)/sum(t_old.^2) > option(1) ) && (iter < option(2))
        temp = sum((t_old - T(:,a)).^2)/sum(t_old.^2)
        iter = iter + 1;
        t_old = T(:,a);
        P(:,a) = X'*T(:,a)/(T(:,a)'*T(:,a));
        P(:,a) = P(:,a)/norm(P(:,a));
        T(:,a) = X*P(:,a)/(P(:,a)'*P(:,a));
    end
    X = X - T(:,a)*P(:,a)';
end

X_res = Xnormal - T*P';
[U1,S1,V1] = svd(X);
[U2,S2,V2] = svd(X_res);