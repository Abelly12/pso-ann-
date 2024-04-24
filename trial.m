clc 
tic 
close all 
rng default 
filename = 'datafile.xlsx'; 
sheetname1 = 'Sheet1'; 
sheetname2 = 'Sheet2'; 
input = xlsread(filename,sheetname1,'A2:E211'); 
target = xlsread(filename,sheetname2,'A2:A211'); 
inputs=input'; 
targets=target'; 
m=length(inputs(:,1)); 
o=length(targets(:,1)); 
n=10; 
net=feedforwardnet(n); 
net=configure(net,inputs,targets); 
kk=m*n+n+n+o; 
for j=1:kk 
LB(1,j)=-1.5; 
UB(1,j)=1.5; 
end 
pop=10; 
for i=1:pop 
for j=1:kk 
xx(i,j)=LB(1,j)+rand*(UB(1,j)-LB(1,j)); 
end 
end 
maxrun=1; 
for run=1:maxrun 
fun=@(x) myfunc(x,n,m,o,net,inputs,targets);
 x0=xx; % pso initialization----------------------------------------------start 
x=x0; 
% initial population 
v=0.1*x0; % initial velocity 
for i=1:pop 
f0(i,1)=fun(x0(i,:)); 
end 
[fmin0,index0]=min(f0);
pbest=x0; % initial pbest 
gbest=x0(index0,:); % initial gbest 
% pso initialization------------------------------------------------end 
% pso algorithm---------------------------------------------------start 
c1=1.5; c2=2.5; 
ite=1; maxite=1000; tolerance=1; 
while ite<=maxite && tolerance>10^-8 
 w=0.1+rand*0.4; 
% pso velocity updates 
 for i=1:pop 
  for j=1:kk 
      v(i,j)=w*v(i,j)+c1*rand*(pbest(i,j)-x(i,j))... 
          +c2*rand*(gbest(1,j)-x(i,j)); 
   end 
 end
 % pso position update
for i=1:pop 
for j=1:kk 
x(i,j)=x(i,j)+v(i,j); 
end 
end 
% handling boundary violations 
for i=1:pop 
for j=1:kk 
if x(i,j)<LB(j) x(i,j)=LB(j); 
elseif x(i,j)>UB(j) x(i,j)=UB(j); 
end 
end 
end 
% evaluating fitness 
for i=1:pop 
f(i,1)=fun(x(i,:)); 
end 
% updating pbest and fitness 
for i=1:pop 
if f(i,1)<f0(i,1) 
pbest(i,:)=x(i,:); 
f0(i,1)=f(i,1); 
end 
end 
[fmin,index]=min(f0); % finding out the best particle 
ffmin(ite,run)=fmin; % storing best fitness 
ffite(run)=ite; % storing iteration count 
% updating gbest and best fitness 
if fmin<fmin0 gbest=pbest(index,:);
 fmin0=fmin; 
end
% calculating tolerance 
if ite>100; 
tolerance=abs(ffmin(ite-100,run)-fmin0); 
end 
% displaying iterative results 
if ite==1 
disp(sprintf('Iteration Best particle Objective fun')); 
end 
 disp(sprintf('%8g %8g %8.4f',ite,index,fmin0)); 
ite=ite+1; 
end % pso algorithm-----------------------------------------------------end 
xo=gbest; 
fval=fun(xo); 
xbest(run,:)=xo; 
ybest(run,1)=fun(xo); 
disp(sprintf('****************************************')); 
disp(sprintf(' RUN fval ObFuVa')); 
disp(sprintf('%6g %6g %8.4f %8.4f',run,fval,ybest(run,1))); 
end 
toc 
% Final neural network model disp('Final nn model is net_f') 

function gmdh = GMDH(params, X, Y)

    disp('Training GMDH:');

    MaxLayerNeurons = params.MaxLayerNeurons;
    MaxLayers = params.MaxLayers;
    alpha = params.alpha;

    nData = size(X,2);
    
    % Shuffle Data
    Permutation = randperm(nData);
    X = X(:,Permutation);
    Y = Y(:,Permutation);
    
    % Divide Data
    pTrainData = params.pTrain;
    nTrainData = round(pTrainData*nData);
    X1 = X(:,1:nTrainData);
    Y1 = Y(:,1:nTrainData);
    pTestData = 1-pTrainData;
    nTestData = nData - nTrainData;
    X2 = X(:,nTrainData+1:end);
    Y2 = Y(:,nTrainData+1:end);
    
    Layers = cell(MaxLayers, 1);

    Z1 = X1;
    Z2 = X2;

    for l = 1:MaxLayers

        L = GetPolynomialLayer(Z1, Y1, Z2, Y2);
        
        if l>1
            if L(1).RMSE2 > Layers{l-1}(1).RMSE2
                break;
            end
        end
        
        ec = alpha*L(1).RMSE2 + (1-alpha)*L(end).RMSE2;
        ec = max(ec, L(1).RMSE2);
        L = L([L.RMSE2] <= ec);
        
        if numel(L) > MaxLayerNeurons
            L = L(1:MaxLayerNeurons);
        end
        
        if l==MaxLayers && numel(L)>1
            L = L(1);
        end

        Layers{l} = L;

        Z1 = reshape([L.Y1hat],nTrainData,[])';
        Z2 = reshape([L.Y2hat],nTestData,[])';

        disp(['Layer ' num2str(l) ': Neurons = ' num2str(numel(L)) ', Min Error = ' num2str(L(1).RMSE2)]);

        if numel(L)==1
            break;
        end

    end

    Layers = Layers(1:l);
    
    gmdh.Layers = Layers;
    
    disp(' ');
    
end
%Calculation of MSE 
err=sum((net_f(inputs)-targets).^2)/length(net_f(inputs));
%Regression plot 
plotregression(targets,net_f(inputs)) 
disp('Trained ANN net_f is ready for the use');