function [w,f,normgrad] = SG_st2(fun,gfun,Y,bsz,w0,samples,a,dr)
%SG Summary of this function goes here
%   Detailed explanation goes here
step = a;
n = size(Y,1);
bsz = min(bsz,n); % batch size
kmax = 8e2; % number of iterations
f = zeros(kmax+1,1);
normgrad = zeros(kmax+1,1);
for smp = 1:samples
    I = randperm(n,bsz);
    f(1) = f(1) + fun(w0);
    normgrad(1) = normgrad(1) + norm(gfun(I,Y,w0));
    w = w0;
    for k = 2:kmax+1
        s = step/(1+(k-2)*dr);
        I = randperm(n,bsz);
        g = gfun(I,Y,w);
        w = w - g*s;
        f(k) = f(k) + fun(w);
        normgrad(k) = normgrad(k) + norm(gfun(I,Y,w));
    end
end
f = f/samples
normgrad = normgrad/samples
end

