function [w,f,normgrad] = SG_st1(fun,gfun,Y,bsz,w,samples,st)
%SG Summary of this function goes here
%   Detailed explanation goes here
step = 0.3;
n = size(Y,1);
bsz = min(bsz,n); % batch size
kmax = 1e3; % number of iterations
f = zeros(kmax+1,1);
normgrad = zeros(kmax+1,1);
for smp = 1:samples
    I = randperm(n,bsz);
    f(1) = f(1) + fun(I,Y,w);
    normgrad(1) = normgrad(1) + norm(gfun(I,Y,w));
    for k = 2:kmax+1
        s = step;
        I = randperm(n,bsz);
        g = gfun(I,Y,w);
        w = w - g*s;
        f(k) = f(k) + fun(I,Y,w);
        normgrad(k) = normgrad(k) + norm(gfun(I,Y,w));
    end
end
f = f/samples
normgrad = normgrad/samples
end

