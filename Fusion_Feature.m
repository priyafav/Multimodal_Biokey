
d1=load('D_ubiris_1.mat');
%d31=load('D_casia.mat');
D=[];
% for i=1:45
%     D(i,:)=[d1.D(i,:),d31.D(i,:)];
% end
for i=1:45
    D(i,:)=[d1.D(i,:)];
end
% Descr_result1=[d1.featuresTrain;d2.featuresTest];
% Descr_result1=load('result.mat');
d4=abs(zscore(D));
d4(isnan(d4)) = 0;
% load('DG.mat');
% d4=DG;
feature1=[];
t=1;l=1;y=[];
for k=1:45
    if(mod(k,5)~=0)
        feature1{t,l}=d4(k,:);
        l=l+1;
    end
    %feature1{t,l}=d4(k,:);
    if(mod(k,5)==0)
        feature1{t,l}=d4(k,:);
        y(k)=t;
        t=t+1;
        l=1;
    end
end
fet=[];
feat=[];
for i=1:9
    fet(:,1:5)=[feature1{i,1}',feature1{i,2}',feature1{i,4}',feature1{i,3}',feature1{i,5}'];
    feat(:,1)=(fet(:,1)-mean(mean(fet,2)));
    feat(:,2)=(fet(:,2)-mean(mean(fet,2)));
    feat(:,3)=(fet(:,3)-mean(mean(fet,2)));
%     feat(:,4)=(fet(:,4)-mean(mean(fet,2)));
%     feat(:,5)=(fet(:,5)-mean(mean(fet,2)));
    
    weight_matrix1=double(cov(feat'));
    %weight_matrix1=feat*feat';
    [V,D] =eigs(weight_matrix1',length(d4));
    [d,ind] = sort(diag(D),'descend');
    d1=abs(d)<=10^-2;
    d3{i}=d1;
    ind1=ind(d1);
     %d1=d;
    Ds = D(ind1,ind1);
    Vs = V(:,ind1);
    model.vs{i}=Vs';
end
projected_feature1=[];
for i=1:9
  for k=1:5
    projected_feature1{i}(:,k)=model.vs{i}*feature1{i,k}';
  end
end
datarest=cell2mat(projected_feature1)';

A = zeros(45,4);
[rows columns] = size(A);
secondColumn = imresize((1:rows/5)', [rows, 1], 'nearest');
A(:, 2) = secondColumn;
%% 
y=secondColumn;
X=(datarest);
[m,n]=size(X);
for i=1:n
    X(:,i)=(X(:,i)-min(X(:,i)))/(max(X(:,i))-min(X(:,i))+eps);
end
diste=[];
for i=1:45
    for j=1:45
        diste(i,j)=1-norm(X(i,:)-X(j,:))/(norm(X(i,:))+norm(X(j,:)));
    end
end

poz=[];
[a,b]=relieff(X,y,5);
ss=1;
for i=1:length(a)
    if b(i)>0
        poz(:,ss)=X(:,i);
        ss=ss+1;
    end
end
cd=[];
W = dist(poz');
W = -W./max(max(W)); % it's a similarity
[lscores] = LaplacianScore(poz, W);
[junk,ind] = sort(-lscores);
tt1=length(ind);
for i=1:tt1
    cd(:,i)=poz(:,ind(i));
end
dist1=[];
for i=1:45
    for j=1:45
        dist1(i,j)=1-norm(cd(i,:)-cd(j,:))/(norm(cd(i,:))+norm(cd(j,:)));
    end
end

y1=y;
%sec=[1,3,5,6,8,10,11,13,15,16,18,20,21,23,25,26,28,30,31,33,35,36,38,40];
sec=1:2:45;
X=[y1 cd];
[mappedX, mapping] = compute_mapping(X(sec,:),'NCA',length(cd));
cd_rca=abs(cd*mapping.M);
distrca=[];
for i=1:45
    for j=1:45
        distrca(i,j)=1-norm(cd_rca(i,:)-cd_rca(j,:))/(norm(cd_rca(i,:))+norm(cd_rca(j,:)));
    end
end