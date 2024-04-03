 function evaluation_info=evaluate_SDSHL(XTrain,YTrain,LTrain,XTest,YTest,LTest,param)
%% Official codes of SDSHL
%%%     Authors                      Teng et al.
%%%
%%%     Title                        Joint Specifics and Dual-Semantic Hashing 
%%%                                     Learning for Cross-Modal Retrieval
%%%
%% Intput
%%%
%%%     XTrain/YTrain                The features of the first/second modality
%%%
%%%     LTrain                       The label set of `XTrain` and `YTrain`
%%%
%%%     XTest/YTest                  The features of the first/second modality
%%%                                  (test sets)
%%%
%%%     LTest                        The ground-truth of test sets
%%%
%% param
%%%
%%%     nbits                        The length of hash codes
%%%
%%%     max_iter                     The number of iteration
%%%
%%%     alpha                        The weight of manifold term
%%%
%%%     k                            The number of clusters/anchors
%%%
%%%     eta1/eta2                    The weight of A^t A^t' in Eq. (6)
%%%
%%%     beta                         The weight of manifold \| A^t - A^p \|
%%%
%%%     gamma                        The weight of regularization w.r.t. A^t,
%%%                                  W^t
%%%
%%%     theta                        The weight of \|G^t-R^t W^t L \|
%%%
%%%     lambda1                      The weight of \| rS - B' G^t \|
%%%
%%%     lambda2                      The weight of \| rS - G^t' G^p \|
%%%
%%%     xi                           The weight of regularization  of Eq. (27)
%%%
%% Output
%%%
%%%     evaluation_info              The result of SDSHL
%%%
%% Version
%%%
%%%     Upload                       2024-04-03
%%%
    warning off;

    %% ----------------- hash learning ----------------------
    tic;
    [B] = train_SDSHL(XTrain',YTrain',LTrain',param);

    XW = (B*XTrain)/(XTrain'*XTrain + param.xi*eye(size(XTrain,2)));
    YW = (B*YTrain)/(YTrain'*YTrain + param.xi*eye(size(YTrain,2)));

    evaluation_info.trainT=toc;

    %% ----------------- codes for query ----------------- 
    BxTrain = compactbit(B'>=0);
    ByTrain = BxTrain;
    BxTest = compactbit((XW*XTest')'>0);
    ByTest = compactbit((YW*YTest')'>0);
    evaluation_info.compressT = toc;
    
    %% ----------------- evaluate ----------------- 
    tic;
    DHamm = hammingDist(BxTest, ByTrain);
    [~, orderH] = sort(DHamm, 2);
	evaluation_info.Image_VS_Text_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Image_VS_Text_precision, evaluation_info.Image_VS_Text_recall] = precision_recall(orderH', LTrain, LTest);
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    [evaluation_info.Text_VS_Image_precision,evaluation_info.Text_VS_Image_recall] = precision_recall(orderH', LTrain, LTest);
    evaluation_info.testT = toc;   
end
