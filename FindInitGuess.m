function [w,l,lcomp] = FindInitGuess(w,A,b)
%FINDINITGUESS Find a initial point to implement Active Set Method
relu = @(w)max(w,0);
drelu = @(w)ones(size(w)).*sign(relu(w));
l = sum(relu(b-A*w));
iter = 0;
itermax = 10000;
step = 0.1;
while l > 0 && iter < itermax
    dl = sum(-drelu(b-A*w)'*A,1)';
    if norm(dl) > 1
        dl = dl/norm(dl);
    end
    w = w - step*dl;
    l = sum(relu(b-A*w));
    iter = iter + 1;
end
lcomp = relu(b-A*w);
end

