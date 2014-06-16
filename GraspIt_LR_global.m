%% GraspIt_LR_global.m
%% 4/28/14
%  Simple fitting of a logistic regression classifier over all data. Used
%  as baseline for LR hybrid.
%  Update 5/23/14: Modified to filter using kdtree and updated data
%%
function GraspIt_LR_global(energycutoff,binThreshold)

addpath('Z:\Thesis\thesisStartup.m');
addpath('Z:\Thesis\thesisCleanup.m');
%me = mfilename;
me = sprintf('kd_global_LR_energy_%d_bin_%d',energycutoff,binThreshold);
use_timestamp = false;
thesisStartup;

% Suppress pesky LR warnings
orig_warn_state = warning;
warning('off','all');

% Number of features to read
dimensionCount = 11;

%AUC Evaluation Parameters
cutoff = .8;        %Anything >= this value is classified as a success

%Data Storage
data = [];          %X parameters for GP Classifier
results1 = [];       %Y parameters for GP Classifier
results2 = [];      %Alternate Y parameter
groundtruth = [];

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
results1str = {};
results2str = {};
res2index = [];


%Shake data collected Jan13
dzzf_human = 1;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    datastr{end+1} = 'zhifei_human_data_post-correction.csv';
end

dzzf_autograsp = 0;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    datastr{end+1} = 'zhifei_autograsp_data_post-correction.csv';
end
   
%------------End Defind Parameters-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Seed Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////


for i = 1:length(datastr)
      datatemp = importdata(datastr{i});
      
      % For these datasets we only want the first 11 metrics
      data = [data;datatemp.data(:,1:dimensionCount)];
      % GraspIt estimate
      results1 = [results1;datatemp.data(:,dimensionCount+1)];
      % No crowdsourced data in the sets
      %results2 = ...
      % Physical testing
      groundtruth = [groundtruth;datatemp.data(:,dimensionCount+3)];
      if i ==1
          header = datatemp.colheaders(2:end);
%      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
%         error('headers do not match, check data files')   
      end   
end

%------------End Seed Data-----------------------------------------

%%
%//////////////////////////////////////////////////////////////////////////
%-------------Condition Datasets------------------------------------
%//////////////////////////////////////////////////////////////////////////

%Remove NAN columns
datatemp = isnan(mean(data));
data(:,datatemp) = [];
disp('!!!!below quality measures have been removed due to insufficient data!!!!!!!!!!!!')
disp( header(datatemp))
header(datatemp) = [];

%%% We want to keep all dimensions so don't delete any
% %Figure out which QM's are significant with T-tests
% [p,keep,delete] = ttester(data,groundtruth,qmset,cutoff);
% disp('Keeping QMs:')
% disp('name        p values')
% for i = 1:length(keep)
%     disp([header(keep(i)) num2str(p(keep(i)),3)] )
% end
% 
% 
% %Delete non-important QM's
% disp('Deleting QMs:')
% disp('name        p values')
% for i = length(delete):-1:1
%     disp([header(delete(i)) num2str(p(delete(i)),3)] )
% end
% header(delete) = [];
% data(:,delete) = [];

data = sphereize(data);

% Run PCA on a copy for plotting the contour
finalDimensionNum = 2;
[~,data_pca,latent] = pca(data);
data_pca = data_pca(:,logical([ones(1,finalDimensionNum) zeros(1,size(data_pca,2)-finalDimensionNum)]));

% Testing only on physical testing successes
physicalCutoff = .8;

% Linear regression prediction must be >= this value to be positive
lrCutoff = .5;

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

loopn = 100;

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .33;
train2Proportion = .33;

%% Build global kdtree
tree = kdMedCreateTree(data,binThreshold);
% Don't split on results, but include them for easy bin confidence
% computing
bins = kdGetAllBins(tree);
fprintf('Number of bins: %d\n',size(bins,1));
bin_size_sum = 0;
for i = 1:size(bins,1)
    bin_size_sum = bin_size_sum + size(bins{i,1},1);
end
fprintf('Average bin size: %d\n',bin_size_sum / size(bins,1));

% Train classifiers
nodeIndices = kdGetAllNodeIndices(tree);

dataNodes = zeros(size(data,1),1);
for i = 1:size(dataNodes,1)
    dataNodes(i) = kdGetBinIndex(tree,data(i,:));
end

summaryStats = [];

% For diagnostic purposes - get the number of positive and negative
% training examples
trainy_total = [];

disp(strcat('energy base:',num2str(energycutoff)));

predictions = {};
trueValues = {};
trueValues_unfiltered = {};

%auc = zeros(1,loopn);
auc = [];
rocx = [];
rocy = [];

% Passed to contour3d - added to each iteration
indexout = [];
mfout = [];

filter_with_confidence = 1;

if(~filter_with_confidence)
    confidence_thresholds = 0;
    binThreshold = size(data,1)+1;
else
    confidence_thresholds = [0 .5:.1:.9];
    %confidence_thresholds = [0 .8];
end

for confidence_cutoff = confidence_thresholds
    predictions = {};
    trueValues = {};
    prediction_points = zeros(loopn,1);
    
    %auc = zeros(1,loopn);
    auc = [];
    rocx = [];
    rocy = [];
    
    % Passed to contour3d - added to each iteration
    mfout = [];
    mfout_precision = [];
    mfout_confidence = [];
    mfout_TP = [];
    indexout = [];
    indexout_precision = [];
    indexout_confidence = [];
    indexout_TP = [];
    
    % Remember precisions for every bin over all runs
    bin_precisions = zeros(loopn,size(bins,1));
    
    l = 1;
    while(l <= loopn)
        trainData = data;
        trainy = (results1 <= energycutoff);
        %trainy = (groundtruth >= cutoff);

        % Make testx vector of appropriate size
        if(train2Proportion >= 1)
            train2x = zeros(train2Proportion,1);
        else
            train2x = zeros(floor(size(trainData,1) * train2Proportion),1);
        end
        if(testProportion >= 1)
            testx = zeros(testProportion,1);
        else
            testx = zeros(floor(size(trainData,1) * testProportion),1);
        end


        train2Indices = randomIndices(trainData,length(train2x));
        testIndices = randomIndices(trainData,length(testx),train2Indices);
        % Unlike most of the other experiments, we want predictions for
        % both test and training points.
        testx = trainData(testIndices,:);
        train2x = trainData(train2Indices,:);
        trainData([testIndices; train2Indices],:) = [];
        trainy([testIndices; train2Indices]) = [];

        trainy_total = [trainy_total; trainy];

        dbstop if error


        % Train classifier
        B = glmfit(trainData, [trainy ones(size(trainy,1),1)], 'binomial', 'link', 'logit');

            % Keeps track of which predictions were from which nodes
            test_containing_nodes = zeros(size(train2Indices,1),1);
            % We want these for all test indices including the eliminated ones
            for i = 1:size(test_containing_nodes,1)
                idx = kdGetBinIndex(tree,data(train2Indices(i),:));
                test_containing_nodes(i) = idx;
            end

            % Classify the train2 points
            mf = Logistic(B(1) + train2x * B(2:end));

            % For every index in nodeIndices:
            % compute the confidence rating by looking at ground truth
            % create logical vector of same size as nodeIndices, saying whether
            % or not to predict in bin.
            confidence = zeros(size(nodeIndices));
            actuals = groundtruth(train2Indices);
            for i = 1:size(nodeIndices,1)
                idx = nodeIndices(i);
                preds = mf(dataNodes(train2Indices) == idx);
                actuals_i = actuals(dataNodes(train2Indices) == idx);
                % conf = TP / (TP + FP)
                TP = sum(preds >= lrCutoff & actuals_i >= physicalCutoff);
                FP = sum(preds >= lrCutoff & actuals_i < physicalCutoff);
                confidence(i) = TP / (TP + FP);
            end

            % For every point in test set:
            % Find kdtree index
            % If confidence for the index is not high, remove from test set
            eliminate = false(size(testx,1),1);
            for i = 1:size(testx,1)
                idx = kdGetBinIndex(tree,data(testIndices(i),:));
                C = confidence(nodeIndices==idx);
                if(confidence_cutoff > 0 && (isnan(C) || C < confidence_cutoff))
                    eliminate(i) = true;
                end
            end

            fprintf('Eliminated %d points\n',sum(eliminate));
            testIndices_final = testIndices(~eliminate);
            %testx(eliminate,:) = [];

            if size(testx,1) < 10
                disp('Not enough valid test points; skipping iteration.');
                continue;
            end

            % For filtered test set:
            % Get LR predictions (this is mf) and ground truth
            mf = zeros(size(testx,1),1);
            % Keeps track of which predictions were from which nodes
            test_containing_nodes = zeros(size(testIndices,1),1);
            % We want these for all test indices including the eliminated ones
            for i = 1:size(test_containing_nodes,1)
                idx = kdGetBinIndex(tree,data(testIndices(i),:));
                test_containing_nodes(i) = idx;
            end
            test_containing_nodes_filtered = test_containing_nodes(~eliminate);

        mf_unfiltered = Logistic(B(1) + testx * B(2:end));
        res = +(groundtruth(testIndices_final) >= cutoff);
        res_unfiltered = +(groundtruth(testIndices) >= cutoff);
        mf = mf_unfiltered(~eliminate);

        mfout = [mfout; mf];
        indexout = [indexout; testIndices_final];

        dbstop if error;    

        % Generate ROC curve
        try
            mf_unfiltered(eliminate) = NaN;
            [perfx,perfy,perft,auc(end+1)] = perfcurve(res_unfiltered',mf_unfiltered',1);
            %rocResults = [rocResults; [perfx perfy]];
            [rocx(:,end+1),rocy(:,end+1)] = rotandproject(perfx,perfy,.01);
        catch err
            auc(l) = NaN;
        end

        disp(strcat('Completed iteration: ',num2str(l)));
        [precision, recall] = precisionAndRecall({mf >= lrCutoff},{res},true,{res_unfiltered});
        disp(strcat('Prediction points:',num2str(size(testIndices_final,1))));
        prediction_points(l) = length(testIndices_final);
        disp('  TP     FP     TN     FN');
        [tp, fp, tn, fn] = getTestResults(mf >= lrCutoff,res);
        disp([tp, fp, tn, fn]);
        disp('Precision / Recall');
        disp([precision recall]);
        predictions{l} = (mf >= lrCutoff);
        trueValues{l} = res;
        trueValues_unfiltered{l} = res_unfiltered;
        
        node_precisions = zeros(size(nodeIndices));
        % Generate the precision of all bins/nodes
        for i = 1:size(nodeIndices,1)
            % This recall value is not accurate. Update the allTestPoints
            % value if recall is needed in the future
            [node_precisions(i), ~] = precisionAndRecall({mf(test_containing_nodes_filtered==nodeIndices(i)) >= lrCutoff},{res(test_containing_nodes_filtered==nodeIndices(i))},true,{res(test_containing_nodes_filtered==nodeIndices(i))});
        end
        bin_precisions(l,:) = node_precisions;
        
        l = l+1;
    end

    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,size(auc,1));
    
    [ tpr,tprerr,tprstd ] = tprextfun( ave,upperstd,uppererr,fpr );
    disp('TPR Info by Lines: FPR Threshold/TPR/TPR STD ERR/TPR STD');
    disp([fpr;tpr;tprerr;tprstd]);
    
    %% Generate overall-precision contour
    contour_file = strcat(savedir,'hybrid_contours\',sprintf('cntr_global_LR_egy_%d_conf_%d_bin_%d',energycutoff,confidence_cutoff*10,binThreshold));
    averageBinPrecisions = mean(bin_precisions)';
    point_precisions = zeros(size(data_pca,1),1);
    for i = 1:length(point_precisions)
        point_precisions(i) = averageBinPrecisions(dataNodes(i) == nodeIndices);
        if point_precisions(i) > .90
            fprintf('index %d: %f, %f\n',i,data_pca(i,1),data_pca(i,2));
        end
    end
    
    %% Dump all high-precision grasp nodes
    fprintf('Printing all grasps falling into high-precision nodes. Conf=%f\n',confidence_cutoff);
    pointInfos = [(1:size(data_pca,1))' data_pca];
    totalPointsDumped = 0;
    for i = 1:size(nodeIndices,1)
        if isnan(averageBinPrecisions(i)) || averageBinPrecisions(i) < .9
            continue;
        end
        nodeDump = pointInfos(dataNodes == nodeIndices(i),:);
        totalPointsDumped = totalPointsDumped+size(nodeDump,1);
        fprintf('Node index %d\n',nodeIndices(i));
        disp(nodeDump);
    end
    fprintf('%d points dumped in total\n',totalPointsDumped);
    clearvars pointInfos nodeDump totalPointsDumped
%     for i = 1:length(point_precisions)
%         point_precisions(i) = averageBinPrecisions(kdGetBinIndex(tree,data(i,:)) == nodeIndices);
%         if point_precisions(i) > .90
%             fprintf('index %d: %f, %f\n',i,data_pca(i,1),data_pca(i,2));
%         end
%     end
    %%
    
    %generateCategorizedContour(data_pca,groundtruth,(1:length(point_precisions))',point_precisions,contour_file,sprintf('Areas of high test precision\nGlobal LR, confidence cutoff=%f',confidence_cutoff),[.5 .75 .9]);
    save(sprintf('%s\\contourdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*10),'data_pca','groundtruth','point_precisions','contour_file','confidence_cutoff');

    save(sprintf('%srocdumps\\%s_conf_%d.mat',savedir,me,confidence_cutoff*10),'uppererr','lowererr','ave');
%     fig = figure;
%     set(fig, 'units', 'inches', 'pos', [8 5 3.25 3])
%     set(gca,'FontSize',10)
%     fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
%     %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
%     hold on
%     plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2)
%     %plot(rocResults(:,1).*100,rocResults(:,2).*100,'k','linewidth',2)
%     plot([0 100],[0 100],'color',[.5 .5 .5])
%     set(gca,'box','on','position',[.17,.15,.78,.8])
%     axis square
%     axis([0 100 0 100])
%     ylabel({'TPR';'(%)'})
%     xlabel('FPR (%)')
%     set(get(gca,'YLabel'),'Rotation',0)
%     plot([40 45],[52,60],'k','linewidth',1)
%     set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
%     text(10,84,'GP with PC1','FontSize',10)
%     text(10,8,'Random Guess','FontSize',10)
    
    disp('Overall results (global LR)');
    if(filter_with_confidence)
        disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
    end
    if(~filter_with_confidence)
        disp('All datapoints used');
    else
        disp('Filtered datapoints');
    end
    fprintf('Prediction points: %f (stddev %f)\n',mean(prediction_points),std(prediction_points));

    %Display AUC Info
    disp('Individual AUCs:');
    auc = auc(~isnan(auc));
    disp(auc');
    fprintf('p-value (Wilcoxon signed ranked test): %f\n',signrank(auc));
    disp('AUC ave err std')
    disp(mean(auc))
    disp(std(auc)/(size(auc,1))^.5)
    disp(std(auc))

    [precision_macro, recall_macro, p_err, r_err] = precisionAndRecall(predictions,trueValues,true,trueValues_unfiltered);
    disp('Precision / Recall macro');
    disp([precision_macro recall_macro]);
    disp('');
    [precision_micro, recall_micro] = precisionAndRecall(predictions,trueValues,false,trueValues_unfiltered);
    disp('Precision / Recall micro');
    disp([precision_micro recall_micro]);
    
    disp('Std devs');
    disp([p_err r_err]);

end


% disp('    TPR       FPR       TNR       FNR       cutoff');
% disp(summaryStats);

warning(orig_warn_state);

thesisCleanup;
end