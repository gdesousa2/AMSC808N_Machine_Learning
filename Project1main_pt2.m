function Project1main_pt2()
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
% ind = find((A(:,1)>=6000 & A(:,1)<=6999)); % ...  %CA
%  | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
%  | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ  
%  | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
%  | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
% A = A(ind,:);

[n,dim] = size(A);

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% select max subset of data with equal numbers of dem and gop counties
ngop = length(igop);
ndem = length(idem);
if ngop > ndem
    rgop = randperm(ngop,ndem);
    Adem = A(idem,:);
    Agop = A(igop(rgop),:);
    A = [Adem;Agop];
else
    rdem = randperm(ndem,ngop);
    Agop = A(igop,:);
    Adem = A(idem(rdem),:);
    A = [Adem;Agop];
end  
[n,dim] = size(A)
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

%% set up data matrix and visualize
%close all
%figure;
%hold on; grid;
X = [A(:,4:9),log(num)];
X(:,1) = X(:,1)/1e4;
% select three data types that distinguish dem and gop counties the most
i1 = 1; % Median Income
i2 = 7; % log(# votes)
i3 = 5; % Bachelor Rate
%plot3(X(idem,i1),X(idem,i2),X(idem,i3),'.','color','b','Markersize',20);
%plot3(X(igop,i1),X(igop,i2),X(igop,i3),'.','color','r','Markersize',20);
%view(3)
%fsz = 16;
%set(gca,'Fontsize',fsz);
%xlabel(str(i1),'Fontsize',fsz);
%ylabel(str(i2),'Fontsize',fsz);
%zlabel(str(i3),'Fontsize',fsz);
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


%% set up optimization problem - Stochastic Gradient Descend - Batch size
step = 0.3;
samp = 1; % samples to average over
bsz = [8; 16; 32; 64; 128; 256; 512];
[n,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w0 = [-1;-1;1;1]; % initial guess
fun = @(w)fun0((1:n)',Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);

t1 = tic;
[wSG1,fSG1,gnormSG1] = SG_st1(fun,gfun,Y,bsz(1),w0,samp,step);
t1 = toc(t1);
t2 = tic;
[wSG2,fSG2,gnormSG2] = SG_st1(fun,gfun,Y,bsz(2),w0,samp,step);
t2 = toc(t2);
t3 = tic;
[wSG3,fSG3,gnormSG3] = SG_st1(fun,gfun,Y,bsz(3),w0,samp,step);
t3 = toc(t3);
t4 = tic;
[wSG4,fSG4,gnormSG4] = SG_st1(fun,gfun,Y,bsz(4),w0,samp,step);
t4 = toc(t4);
t5 = tic;
[wSG5,fSG5,gnormSG5] = SG_st1(fun,gfun,Y,bsz(5),w0,samp,step);
t5 = toc(t5);
t6 = tic;
[wSG6,fSG6,gnormSG6] = SG_st1(fun,gfun,Y,bsz(6),w0,samp,step);
t6 = toc(t6);
t7 = tic;
[wSG7,fSG7,gnormSG7] = SG_st1(fun,gfun,Y,bsz(7),w0,samp,step);
t7 = toc(t7);

runtime = [t1; t2; t3; t4; t5; t6; t7]; % run time for different batch sizes

fprintf('wSG1 = [%d,%d,%d], b = %d\n',wSG1(1),wSG1(2),wSG1(3),wSG1(4));
fprintf('wSG2 = [%d,%d,%d], b = %d\n',wSG2(1),wSG2(2),wSG2(3),wSG2(4));
fprintf('wSG3 = [%d,%d,%d], b = %d\n',wSG3(1),wSG3(2),wSG3(3),wSG3(4));
fprintf('wSG4 = [%d,%d,%d], b = %d\n',wSG4(1),wSG4(2),wSG4(3),wSG4(4));
fprintf('wSG5 = [%d,%d,%d], b = %d\n',wSG5(1),wSG5(2),wSG5(3),wSG5(4));
fprintf('wSG6 = [%d,%d,%d], b = %d\n',wSG6(1),wSG6(2),wSG6(3),wSG6(4));
fprintf('wSG7 = [%d,%d,%d], b = %d\n',wSG7(1),wSG7(2),wSG7(3),wSG7(4));

% plotting
figure;
hold on; grid;
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);

plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = wSG7(1)*xx+wSG7(2)*yy+wSG7(3)*zz+wSG7(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = '#FFA500';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

figure;
hold on;
grid;
niter = length(f);
plot((0:niter-1)',f,'Linewidth',2,'Color','k','DisplayName','SINewton');
niterSG = length(fSG1);
plot((0:niterSG-1)',fSG1,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(1))));
plot((0:niterSG-1)',fSG2,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(2))));
plot((0:niterSG-1)',fSG3,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(3))));
plot((0:niterSG-1)',fSG4,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(4))));
plot((0:niterSG-1)',fSG5,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(5))));
plot((0:niterSG-1)',fSG6,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(6))));
plot((0:niterSG-1)',fSG7,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(7))));
set(gca,'Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
legend();

figure;
hold on;
grid;
niter = length(gnorm);
plot((0:niter-1)',gnorm,'Linewidth',2,'Color','k','DisplayName','SINewton');
niterSG = length(gnormSG1);
plot((0:niterSG-1)',gnormSG1,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(1))));
plot((0:niterSG-1)',gnormSG2,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(2))));
plot((0:niterSG-1)',gnormSG3,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(3))));
plot((0:niterSG-1)',gnormSG4,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(4))));
plot((0:niterSG-1)',gnormSG5,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(5))));
plot((0:niterSG-1)',gnormSG6,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(6))));
plot((0:niterSG-1)',gnormSG7,'Linewidth',2,'DisplayName',strcat('bsz = ',num2str(bsz(7))));
set(gca,'Fontsize',fsz);
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);
legend();

figure;
hold on;
grid;
plot(bsz,runtime)
set(gca,'Fontsize',fsz);
xlabel('bsz','Fontsize',fsz);
ylabel('runtime (s)','Fontsize',fsz);


%% set up optimization problem - Stochastic Gradient Descend - Decreasing strategies
step = [0.1; 0.3; 0.5]; % step size - strategy 1
dec_rate = [0.01; 0.1; 1;]; % decay rate - strategy 2
samp = 1; % samples to average over
bsz = 64;
[n,dim] = size(XX);
lam = 0.01;
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w0 = [-1;-1;1;1]; % initial guess
fun = @(w)fun0((1:n)',Y,w,lam);
gfun = @(I,Y,w)gfun0(I,Y,w,lam);

t1 = tic;
[wST1,fST1,gnormST1] = SG_st1(fun,gfun,Y,bsz,w0,samp,step(1));
t1 = toc(t1);
t2 = tic;
[wST2,fST2,gnormST2] = SG_st1(fun,gfun,Y,bsz,w0,samp,step(2));
t2 = toc(t2);
t3 = tic;
[wST3,fST3,gnormST3] = SG_st1(fun,gfun,Y,bsz,w0,samp,step(3));
t3 = toc(t3);
t4 = tic;
[wST4,fST4,gnormST4] = SG_st2(fun,gfun,Y,bsz,w0,samp,step(2),dec_rate(1));
t4 = toc(t4);
t5 = tic;
[wST5,fST5,gnormST5] = SG_st2(fun,gfun,Y,bsz,w0,samp,step(2),dec_rate(1));
t5 = toc(t5);
t6 = tic;
[wST6,fST6,gnormST6] = SG_st2(fun,gfun,Y,bsz,w0,samp,step(2),dec_rate(2));
t6 = toc(t6);
t7 = tic;
[wST7,fST7,gnormST7] = SG_st3(fun,gfun,Y,bsz,w0,samp);
t7 = toc(t7);

runtime = [t1; t2; t3; t4; t5; t6; t7]; % run time for different batch sizes

fprintf('wST1 = [%d,%d,%d], b = %d\n',wST1(1),wST1(2),wST1(3),wST1(4));
fprintf('wST2 = [%d,%d,%d], b = %d\n',wST2(1),wST2(2),wST2(3),wST2(4));
fprintf('wST3 = [%d,%d,%d], b = %d\n',wST3(1),wST3(2),wST3(3),wST3(4));
fprintf('wST4 = [%d,%d,%d], b = %d\n',wST4(1),wST4(2),wST4(3),wST4(4));
fprintf('wST5 = [%d,%d,%d], b = %d\n',wST5(1),wST5(2),wST5(3),wST5(4));
fprintf('wST6 = [%d,%d,%d], b = %d\n',wST6(1),wST6(2),wST6(3),wST6(4));
fprintf('wST7 = [%d,%d,%d], b = %d\n',wST7(1),wST7(2),wST7(3),wST7(4));

% plotting
figure;
hold on; grid;
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);

plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = wST3(1)*xx+wST3(2)*yy+wST3(3)*zz+wST3(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = '#FFA500';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

figure;
hold on;
grid;
niter = length(f);
plot((0:niter-1)',f,'Linewidth',2,'Color','k','DisplayName','SINewton');
niterST = length(fST1);
plot((0:niterST-1)',fST1,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(1))));
plot((0:niterST-1)',fST2,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(2))));
plot((0:niterST-1)',fST3,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3))));
plot((0:niterST-1)',fST4,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(2)),...
                                                    ', \gamma = ',num2str(dec_rate(1))));
plot((0:niterST-1)',fST5,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(2)),...
                                                    ', \gamma = ',num2str(dec_rate(2))));
plot((0:niterST-1)',fST6,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(2)),...
                                                    ', \gamma = ',num2str(dec_rate(3))));
plot((0:niterST-1)',fST7,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3)),', ls'));
set(gca,'Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
legend();

figure;
hold on;
grid;
niter = length(gnorm);
plot((0:niter-1)',gnorm,'Linewidth',2,'Color','k','DisplayName','SINewton');
niterST = length(gnormST1);
plot((0:niterST-1)',gnormST1,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(1))));
plot((0:niterST-1)',gnormST2,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(2))));
plot((0:niterST-1)',gnormST3,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3))));
plot((0:niterST-1)',gnormST4,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3)),...
                                                    ', \gamma = ',num2str(dec_rate(1))));
plot((0:niterST-1)',gnormST5,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3)),...
                                                    ', \gamma = ',num2str(dec_rate(2))));
plot((0:niterST-1)',gnormST6,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3)),...
                                                    ', \gamma = ',num2str(dec_rate(3))));
plot((0:niterST-1)',gnormST7,'Linewidth',2,'DisplayName',strcat('\alpha = ',num2str(step(3)),', ls'));
set(gca,'Fontsize',fsz);
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);
legend();

figure;
hold on;
grid;
plot(runtime,'^')
set(gca,'Fontsize',fsz);
xlabel('strategies','Fontsize',fsz);
xticks([1 2 3 4 5 6 7]);
xticklabels({'\alpha = 0.1'; '\alpha = 0.3'; '\alpha = 0.5'; '\gamma = 0.5'; '\gamma = 1'; '\gamma = 2'; 'ls'});
ylabel('runtime (s)','Fontsize',fsz);
set(gca,'XTickLabelRotation',45)

%% set up optimization problem - Stochastic Gradient Descend - Tickonov regularization
step = 0.3;
samp = 1; % samples to average over
bsz = 64;
[n,dim] = size(XX);
lam = [0.0001, 0.001, 0.01, 0.02, 0.03];
Y = (label*ones(1,dim + 1)).*[XX,ones(n,1)];
w0 = [-1;-1;1;1]; % initial guess

t1 = tic;
[wLB1,fLB1,gnormLB1] = SG_st1(@(w)fun0((1:n)',Y,w,lam(1)),@(I,Y,w)gfun0(I,Y,w,lam(1)),Y,bsz(1),w0,samp,step);
t1 = toc(t1);
t2 = tic;
[wLB2,fLB2,gnormLB2] = SG_st1(@(w)fun0((1:n)',Y,w,lam(2)),@(I,Y,w)gfun0(I,Y,w,lam(2)),Y,bsz(1),w0,samp,step);
t2 = toc(t2);
t3 = tic;
[wLB3,fLB3,gnormLB3] = SG_st1(@(w)fun0((1:n)',Y,w,lam(3)),@(I,Y,w)gfun0(I,Y,w,lam(3)),Y,bsz(1),w0,samp,step);
t3 = toc(t3);
t4 = tic;
[wLB4,fLB4,gnormLB4] = SG_st1(@(w)fun0((1:n)',Y,w,lam(4)),@(I,Y,w)gfun0(I,Y,w,lam(4)),Y,bsz(1),w0,samp,step);
t4 = toc(t4);
t5 = tic;
[wLB5,fLB5,gnormLB5] = SG_st1(@(w)fun0((1:n)',Y,w,lam(5)),@(I,Y,w)gfun0(I,Y,w,lam(5)),Y,bsz(1),w0,samp,step);
t5 = toc(t5);

runtime = [t1; t2; t3; t4; t5]; % run time for different batch sizes

fprintf('wLB1 = [%d,%d,%d], b = %d\n',wLB1(1),wLB1(2),wLB1(3),wLB1(4));
fprintf('wLB2 = [%d,%d,%d], b = %d\n',wLB2(1),wLB2(2),wLB2(3),wLB2(4));
fprintf('wLB3 = [%d,%d,%d], b = %d\n',wLB3(1),wLB3(2),wLB3(3),wLB3(4));
fprintf('wLB4 = [%d,%d,%d], b = %d\n',wLB4(1),wLB4(2),wLB4(3),wLB4(4));
fprintf('wLB5 = [%d,%d,%d], b = %d\n',wLB5(1),wLB5(2),wLB5(3),wLB5(4));

% plotting
figure;
hold on; grid;
plot3(XX(idem,1),XX(idem,2),XX(idem,3),'.','color','b','Markersize',20);
plot3(XX(igop,1),XX(igop,2),XX(igop,3),'.','color','r','Markersize',20);
view(3)
fsz = 16;
set(gca,'Fontsize',fsz);
xlabel(str(i1),'Fontsize',fsz);
ylabel(str(i2),'Fontsize',fsz);
zlabel(str(i3),'Fontsize',fsz);

plane = w(1)*xx+w(2)*yy+w(3)*zz+w(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = 'green';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

xmin = min(XX(:,1)); xmax = max(XX(:,1));
ymin = min(XX(:,2)); ymax = max(XX(:,2));
zmin = min(XX(:,3)); zmax = max(XX(:,3));
nn = 50;
[xx,yy,zz] = meshgrid(linspace(xmin,xmax,nn),linspace(ymin,ymax,nn),...
    linspace(zmin,zmax,nn));
plane = wLB1(1)*xx+wLB1(2)*yy+wLB1(3)*zz+wLB1(4);
p = patch(isosurface(xx,yy,zz,plane,0));
p.FaceColor = '#FFA500';
p.EdgeColor = 'none';
camlight 
lighting gouraud
alpha(0.3);

figure;
hold on;
grid;
niter = length(f);
plot((0:niter-1)',f,'Linewidth',2,'Color','k','DisplayName','SINewton');
niterLB = length(fLB1);
plot((0:niterLB-1)',fLB1,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(1))));
plot((0:niterLB-1)',fLB2,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(2))));
plot((0:niterLB-1)',fLB3,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(3))));
plot((0:niterLB-1)',fLB4,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(4))));
plot((0:niterLB-1)',fLB5,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(5))));
set(gca,'Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
legend();

figure;
hold on;
grid;
niter = length(gnorm);
plot((0:niter-1)',gnorm,'Linewidth',2,'Color','k','DisplayName','SINewton');
niterLB = length(gnormLB1);
plot((0:niterLB-1)',gnormLB1,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(1))));
plot((0:niterLB-1)',gnormLB2,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(2))));
plot((0:niterLB-1)',gnormLB3,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(3))));
plot((0:niterLB-1)',gnormLB4,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(4))));
plot((0:niterLB-1)',gnormLB5,'Linewidth',2,'DisplayName',strcat('\lambda = ',num2str(lam(5))));
set(gca,'Fontsize',fsz);
set(gca,'YScale','log');
xlabel('k','Fontsize',fsz);
ylabel('|| stoch grad f||','Fontsize',fsz);
legend();

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
end








