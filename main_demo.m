clear;clc
addpath(genpath('./utils/'));
addpath(genpath('./main/'));
result_URL = './results/';
if ~isfolder(result_URL)
    mkdir(result_URL);
end
db = {'NUSWIDE'};
hashmethods = {'SDSHL'};
loopnbits = [8 16 32 64 96 128]; 
% params setting
param.k = 61;
param.alpha = 10 ;
param.beta = 10;
param.lambda1 = 10;
param.lambda2 = 100;
param.theta = 1;
param.gamma = 1;
param.xi = 10;
param.eta1 = 0.4; 
param.eta2 = 1-param.eta1;
param.max_iter = 2;
kSelect = cell(length(db),2);
for dbi = 1:length(db)
    db_name = db{dbi}; 
    param.db_name = db_name;
    % load dataset
    load(['./datasets/',db_name,'.mat']);
    result_name = [result_URL 'SDSHL_' db_name '_result' '.mat'];
    XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
    XTest = I_te; YTest = T_te; LTest = L_te;
    clear X Y L I_tr I_te T_tr T_te L_tr L_te 
    %% Format
    if isvector(LTrain)
        LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
        LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
    end
    %% Methods
    eva_info = cell(length(hashmethods),length(loopnbits));
    seed = 2022;
    rng(seed);
    for ii =1:length(loopnbits)
        fprintf('======%s: start %d bits encoding======\n\n',db_name,loopnbits(ii)); 
        param.nbits = loopnbits(ii); 
        for jj = 1:length(hashmethods)
            eva_info_ = evaluate_SDSHL(XTrain,YTrain,LTrain,XTest,YTest,LTest,param);
            eva_info{jj,ii} = eva_info_;
            clear eva_info_
        end
    end
    % MAP
    for ii = 1:length(loopnbits)
        for jj = 1:length(hashmethods)
            % MAP
            Image_VS_Text_MAP{jj,ii} = eva_info{jj,ii}.Image_VS_Text_MAP;
            Text_VS_Image_MAP{jj,ii} = eva_info{jj,ii}.Text_VS_Image_MAP;
            % time
            trainT{jj,ii} = eva_info{jj,ii}.trainT;
            compressT{jj,ii} = eva_info{jj,ii}.compressT;
            testT{jj,ii} = eva_info{jj,ii}.testT;
        end
        fprintf("%dbits  I2T = %f ; T2I = %f ;      trainT = %f\n",loopnbits(ii),Image_VS_Text_MAP{jj,ii},Text_VS_Image_MAP{jj,ii},trainT{jj,ii});
    end
    % I -> T
    % T -> I
    % Training time
    % Query time
    save(result_name,'Image_VS_Text_MAP','Text_VS_Image_MAP','trainT','testT');
end