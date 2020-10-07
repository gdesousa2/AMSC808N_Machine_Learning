function [x,f,gnorm] = SLBFGS(func,gfun,Y,bsz,x0,m)
gam = 0.9; % line search step factor
jmax = ceil(log(1e-14)/log(gam)); % max # of iterations in line search
eta = 0.5; % backtracking stopping criterion factor
tol = 1e-10;
m = m; % the number of steps to keep in memory
dim = max(size(x0,1),size(x0,2));
n = size(Y,1);
bsz = min(n,bsz);
itermax = 1000;
%% 
s = zeros(dim,m);
y = zeros(dim,m);
rho = zeros(1,m);
f = zeros(1,itermax);
gnorm = zeros(1,itermax);
%
x = x0;
f(1) = [func(x)];
I = randperm(n,bsz);
g = gfun(I,Y,x);
gnorm(1) = norm(g);
% first do steepest decend step
a = linesearch(x,-g,g,func,eta,gam,jmax);
xnew = x - a*g;
f(2) = func(xnew);
I = randperm(n,bsz);
gnew = gfun(I,Y,xnew);
s(:,1) = xnew - x;
y(:,1) = gnew - g;
rho(1) = 1/(s(:,1)'*y(:,1));
x = xnew;
g = gnew;
nor = norm(g);
gnorm(2) = nor;
iter = 1;
while (nor > tol) && (iter < itermax)
    if iter < m
        I = 1 : iter;
        p = finddirection(g,s(:,I),y(:,I),rho(I));
    else
        p = finddirection(g,s,y,rho);
    end
    [a,j] = linesearch(x,p,g,func,eta,gam,jmax);
    if j == jmax
        p = -g;
        [a,j] = linesearch(x,p,g,func,eta,gam,jmax);
    end
    I = randperm(n,bsz);
    step = a*p;
    xnew = x + step;
    gnew = gfun(I,Y,xnew);
    s = circshift(s,[0,1]); 
    y = circshift(y,[0,1]);
    rho = circshift(rho,[0,1]);
    s(:,1) = step;
    y(:,1) = gnew - g;
    rho(1) = 1/(step'*y(:,1));
    x = xnew;
    g = gnew;
    nor = norm(g);
    iter = iter + 1;
    gnorm(iter+1) = nor;
    f(iter+1) = func(x);
end
fprintf('S L-BFGS: %d iterations, norm(g) = %d\n',iter,nor);
gnorm(iter+1:end) = [];
f(iter+1:end) = [];
end

%%
function [a,j] = linesearch(x,p,g,func,eta,gam,jmax)
    a = 1;
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