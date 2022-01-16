
%LOCAL-GLOBAL feature extraction
D=[];
for k=1:9
    for s=1:8
  descList = {'BPPC','GDP','GDP2','GLTP','IWBC',...
            'LAP','LBP','LDiP','LDiPv','LDN',...
            'LDTP','LFD','LGBPHS','LGDiP','LGIP',...
            'LGP','LGTrP','LMP','LPQ','LTeP',...
            'LTrP','MBC','MBP','MRELBP','MTP',...
            'mWLD','PHOG'};
options.gridHist = 1;
de=27;%%%%Local ternary pattern
desc = descList{de};
descFunc = str2func(['desc_' desc]);  
  
  InImage=(imread([num2str(k) '_' num2str(s) '.bmp']));
  %InImage=(imread(['10' num2str(k) '_' num2str(s) 'Roi_bc_pca_fdb100.tif']));
     if size(InImage,3) == 3
        InImage = rgb2gray(InImage);
    end
    disp('recognizing part : 1')
    
%   %part2: apply gabor filter
    %create a gabor array of scale 3 and orientation 3
    gaborArray = GaborFilterBank();
    [cA,cH,cV,cD] = dwt2(InImage,'sym4','mode','per');
    %apply gabor filter bank to image and assign the output images to gaborResult
%     gaborResult = GaborFeatures(cA,gaborArray);
     C=imresize(cA,[64,64]);
    % imwrite(uint8(C),'resize.png');
[gaborResult, gaborResult1] = gaborconvolve(cA,  5, 8, 3, 1.7, ...
		   0.65, 1.3,0, 0);
    disp('recognizing part : 2')
 %   imwrite(gaborResult{1,1},'res1.png');
   % imwrite(gaborResult{1,2},'res2.png');
  %  imwrite(gaborResult{5,8},'res3.png');
%   %part3: apply zernike moment
    %feature vector
    HuResult = cell(1,40);
    ZernikeResult=cell(1,40);
    index = 1;
    
    % get result of zernike moment
    for i = 1:5
        for j = 1:8
            %each one of 9 gabor image is used for zerike
             [~, A1, ~] = Zernikemoment(real(cell2mat(gaborResult(i,j))));
             eta_mat = SI_Moment(real(cell2mat(gaborResult(i,j))));
             A = Hu_Moments(eta_mat);
            ZernikeResult{index} = A1;
            HuResult{index}=A;
            index = index +1;
        end
    end
    disp('recognizing part : 3')
    
%   %part4: apply HOG
    %get number of pixels row and column wise
    [row, column, ~] = size(InImage);
    rows = floor(row/3);
    columns = floor(column/3);
    
    %desired cell size, block size and bins 
    cellsize = [rows columns];
    blocksize = [1 1];
    bin = 9;
   [featureVector, imgDesc] = descFunc(double(InImage),options);
    %extract HOG features and store in global variable
%     [featureVector, ~] = extractHOGFeatures(InImage,'cellsize',cellsize,'BlockSize',blocksize,'NumBins',bin);
    
    %convert data type from single to double
    featureVector = double(featureVector);
    
    %local feature vector generated
    hogResult = num2cell(featureVector);
    disp('recognizing part : 4')
    
%   %part5: apply Nearest neighbour
    %generate ggz feature    
    ggzFeature = cell(1,248);
    
    for i = 1:40
        ggzFeature{i} =ZernikeResult{i};
    end
    
    for i = 41:80
        ggzFeature{i} =HuResult{i-40};
    end
    for j = 81:168+80
        ggzFeature{j} = hogResult{j-80};
    end
    
    %convert cell to matrix type
    ggzFeature = cell2mat(ggzFeature);
    D=[D;ggzFeature];
    end
end