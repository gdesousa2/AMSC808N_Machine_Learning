function Project1main_pt4()
close all
%% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
"Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];
%
% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
%% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];
%% select CA, OR, WA, NJ, NY counties
 ind = find((A(:,1)>=6000 & A(:,1)<=6999)); % ...  %CA
%  | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
 A = A(ind,:);

[n,dim] = size(A);

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% select max subset of data with equal numbers of dem and gop counties
%ngop = length(igop);
%ndem = length(idem);
%if ngop > ndem
%    rgop = randperm(ngop,ndem);
%    Adem = A(idem,:);
%    Agop = A(igop(rgop),:);
%    A = [Adem;Agop];
%else
%    rdem = randperm(ndem,ngop);
%    Agop = A(igop,:);
%    Adem = A(idem(rdem),:);
%    A = [Adem;Agop];
%end  
%[n,dim] = size(A)
%idem = find(A(:,2) >= A(:,3));
%igop = find(A(:,2) < A(:,3));
%num = A(:,2)+A(:,3);
%label = zeros(n,1);
%label(idem) = -1;
%label(igop) = 1;

%% set up data matrix and visualize
close all
figure;
hold on; grid;
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
plot3(X(idem,i1),X(idem,i2),X(idem,i3),'.','color','b','Markersize',20);
plot3(X(igop,i1),X(igop,i2),X(igop,i3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% rescale data to [0,1] and visualize
figure;
hold on; grid;
XX = X(:,[i1,i2,i3]); % data matrix
% rescale all data to [0,1]
xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
X1 = (XX(:,1)-xmin)/(xmax-xmin);
X2 = (XX(:,2)-ymin)/(ymax-ymin);
X3 = (XX(:,3)-zmin)/(zmax-zmin);
XX = [X1,X2,X3];
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);
%% set up optimization problem
[n,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w = [-1;-1;1;1];
fun = @(I,Y,w)fun0(I,Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);
Hvec = @(I,Y,w,v)Hvec0(I,Y,w,v,lam);

[w,f,gnorm] = SINewton(fun,gfun,Hvec,Y,w,64);

fprintf('w = [%d,%d,%d], b = %d\n',w(1),w(2),w(3),w(4));

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

%% set up optimization problem - L-BFGS - Guilherme
m = 5; %memory size
bsz = [16, 32, 64];
[n,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w0 = FindInitGuess(w,Y,ones(n,1)); % initial guess
fun = @(w)fun0((1:n)',Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);

[w1,f1,gnorm1] = SLBFGS(fun,gfun,Y,bsz(1),w0,m);
[w2,f2,gnorm2] = SLBFGS(fun,gfun,Y,bsz(2),w0,m);
[w3,f3,gnorm3] = SLBFGS(fun,gfun,Y,bsz(3),w0,m);

fprintf('w1 = [%d,%d,%d], b = %d\n',w1(1),w1(2),w1(3),w1(4));
fprintf('w2 = [%d,%d,%d], b = %d\n',w2(1),w2(2),w2(3),w2(4));
fprintf('w3 = [%d,%d,%d], b = %d\n',w3(1),w3(2),w3(3),w3(4));

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = w1(1)*xx+w1(2)*yy+w1(3)*zz+w1(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = '#FFA500';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

plane = w2(1)*xx+w2(2)*yy+w2(3)*zz+w2(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = '#FFA500';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

plane = w3(1)*xx+w3(2)*yy+w3(3)*zz+w3(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = '#800080';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);
%%
figure;
hold on;
grid;
niter = length(f);
niter1 = length(f1);
niter2 = length(f2);
niter3 = length(f3);
plot((0:niter-1)',f,'Linewidth',2,'DisplayName','SG');
plot((0:niter1-1)',f1,'Linewidth',2,'DisplayName',strcat('S LBFGS - bsz = ',num2str(bsz(1))));
plot((0:niter2-1)',f2,'Linewidth',2,'DisplayName',strcat('S LBFGS - bsz = ',num2str(bsz(2))));
plot((0:niter3-1)',f3,'Linewidth',2,'DisplayName',strcat('S LBFGS - bsz = ',num2str(bsz(3))));
set(gca,'Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
xlim([0 50])
legend();
%%
figure;
hold on;
grid;
niter = length(gnorm);
niter1 = length(gnorm1);
niter2 = length(gnorm2);
niter3 = length(gnorm3);
plot((0:niter-1)',gnorm,'Linewidth',2,'DisplayName','SG');
plot((0:niter1-1)',gnorm1,'Linewidth',2,'DisplayName',strcat('S LBFGS - bsz = ',num2str(bsz(1))));
plot((0:niter2-1)',gnorm2,'Linewidth',2,'DisplayName',strcat('S LBFGS - bsz = ',num2str(bsz(2))));
plot((0:niter3-1)',gnorm3,'Linewidth',2,'DisplayName',strcat('S LBFGS - bsz = ',num2str(bsz(3))));
set(gca,'Fontsize',fsz);
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);
xlim([0 50])
legend();
end
%%
function f = fun0(I,Y,w,lam)
f = sum(log(1 + exp(-Y(I,:)*w)))/length(I) + 0.5*lam*w'*w;
end
%%
function g = gfun0(I,Y,w,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
g = sum(-Y(I,:).*((aux./(1 + aux))*ones(1,d1)),1)'/length(I) + lam*w;
end
%%
function Hv = Hvec0(I,Y,w,v,lam)
aux = exp(-Y(I,:)*w);
d1 = size(Y,2);
Hv = sum(Y(I,:).*((aux.*(Y(I,:)*v)./((1+aux).^2)).*ones(1,d1)),1)' + lam*v;
end









