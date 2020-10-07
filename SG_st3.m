function [w,f,normgrad] = SG_st3(fun,gfun,Y,bsz,w0,samples)
%SG Summary of this function goes here
%   Detailed explanation goes here
gam = 0.9; % line search step factor
jmax = ceil(log(5e-1)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
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
        I = randperm(n,bsz);
        g = gfun(I,Y,w);
        s = linesearch(w,-g,g,fun,eta,gam,jmax);
        w = w - g*s;
        f(k) = f(k) + fun(w);
        normgrad(k) = normgrad(k) + norm(gfun(I,Y,w));
    end
end
f = f/samples
normgrad = normgrad/samples
end

%%
function [a,j] = linesearch(x,p,g,func,eta,gam,jmax)
    a = 0.3;
    %f0 = func(x(1),x(2));
    f0 = func(x);
    aux = eta*g'*p;
    for j = 0 : jmax
        xtry = x + a*p;
        %f1 = func(xtry(1),xtry(2));
        f1 = func(xtry);
        if f1 < f0 + a*aux
            break;
        else
            a = a*gam;
        end
    end
end