function [evaluation_info] = calcMap(B,param)

    XTrain = param.XTrain;
    YTrain = param.YTrain;
    LTrain = param.LTrain;

    XTest = param.XTest;
    YTest = param.YTest;
    LTest = param.LTest;

   % 学习哈希函数
    XW = (B*XTrain)/(XTrain'*XTrain + param.xi*eye(size(XTrain,2)));
    YW = (B*YTrain)/(YTrain'*YTrain + param.xi*eye(size(YTrain,2)));
    evaluation_info.trainT=toc;

    %% ----------------- codes for query ----------------- 
    tic;
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
    
    DHamm = hammingDist(ByTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);
    
    evaluation_info.testT = toc; 
end