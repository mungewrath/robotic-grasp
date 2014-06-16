%% GP_GraspIt_TP_10d_hybrid_kdtree.m
%% 5/4/14
%  GP classifier trained on 10-dimensional energy data. Used as baseline
%  for Logistic Regression classifier.
%%
function GP_GraspIt_TP_10d_hybrid_kdtree(energycutoff,binThreshold)

addpath('Z:\Thesis\thesisStartup.m');
addpath('Z:\Thesis\thesisCleanup.m');
me = mfilename;
thesisStartup;

% Number of features to read
dimensionCount = 11;

% Number of PCs to keep
finalDimensionNum = 10;

%AUC Evaluation Parameters
cutoff = .8;        %Anything >= this value is classified as a success

%Data Storage
data = [];          %X parameters for GP Classifier
results1 = [];       %Y parameters for GP Classifier
results2 = [];      %Alternate Y parameter
groundtruth = [];

binCount = 10;

%---------------------------------------------------------------------------
%Data Conditions (Value of 1 turns on, value of 0 turns off unless specified)
%---------------------------------------------------------------------------

datastr = {};
results1str = {};
results2str = {};
res2index = [];


filter_with_confidence = 1; % <<< Set to 1 to predict only in areas of confidence.


%Shake data collected Jan13
dzzf_human = 1;% <<<TURN ON HERE<<<<<
if dzzf_human > 0   
    datastr{end+1} = 'zhifei_human_data.csv';
end

dzzf_autograsp = 1;% <<<TURN ON HERE<<<<<
if dzzf_autograsp > 0   
    datastr{end+1} = 'zhifei_autograsp_data_sanitized.csv';
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
      elseif i>1 && mean(strcmp(datatemp.colheaders(2:end),header))<1
         error('headers do not match, check data files')   
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

%Sphereize data
data = sphereize(data);

% Run PCA on a copy for plotting the contour
reducedDimNum = 2;
[~,data_pca,latent] = pca(data);

%Reduce components based on earlier specifications
data_pca = data_pca(:,logical([ones(1,reducedDimNum) zeros(1,size(data,2)-reducedDimNum)]));

if comptype ==1
    
    %Re-distribute using component analysis
    [~,data,latent] = pca(data);

    %Reduce components based on earlier specifications
    data = data(:,logical([ones(1,finalDimensionNum) zeros(1,size(data,2)-finalDimensionNum)]));
end

% Testing only on physical testing successes
physicalCutoff = .8;

contourPrecisionCutoff = .5;

% GP prediction must be higher than this value to be considered a success
gpCutoff = .5;

% data = data((groundtruth > physicalCutoff),:);
% groundtruth = groundtruth(groundtruth > physicalCutoff);
% results1 = results1(groundtruth > physicalCutoff);
% results2 = results2(groundtruth > physicalCutoff);

%------------End condition Data-----------------------------------------
% ----- End copied code ----- %

loopn = 30;

% Keep with cutoff of 0.8
%binshakeresults = bincutoff(groundtruth,0.8);

% Data to use as testing set. If < 1, it is interpreted as
% a percentage; if >= 1, selects this many data points for the set.
testProportion = .33;
train2Proportion = .33;

summaryStats = [];

meanfunc = @meanConst;
covfunc = @covSEard;
likfunc = @likGauss;
inffunc = @infExact;


% For n iterations:
% Train GP on energy
% Obtain predictions based on 60/20/20 split
% Determine areas of PCA space with high TP rates (TP / FP > threshold)
% Add 1 to each TPR,FPR,TNR,FNR based on physical testing
% 
% Aggregate all rates. Output each based on highest count occurrence
% for each, and export
%
% Run testing on last set and record AUC values
% end iterations

% For diagnostic purposes - get the number of positive and negative
% training examples
trainy_total = [];

% A bin needs at least this many points to be considered confident
support_threshold = 3;
% Maximum number of elements per bin for kd tree
%binThreshold = 10; %<<<<<<<< UNCOMMENT AFTER RUNNING BATCH
disp(strcat('binThreshold:',num2str(binThreshold)));
disp(strcat('energy base:',num2str(energycutoff)));
disp(strcat('using confidence:',num2str(filter_with_confidence)));

if(~filter_with_confidence)
    confidence_thresholds = 0;
else
    confidence_thresholds = .5:.1:.9;
end
% Subtract/add a small number to prevent minimum value being placed in
% bin_index 0 and maximum in 11.
PCA_min = min(data)-.0001;
PCA_max = max(data)+.0002;
for confidence_cutoff = confidence_thresholds
    predictions = {};
    trueValues = {};
    
    %auc = zeros(1,loopn);
    auc = [];
    rocx = [];
    rocy = [];
    
    coeff_linear = [];
    coeff_const = [];
    
    TPR = zeros(size(data,1),1);
    FPR = zeros(size(data,1),1);
    TNR = zeros(size(data,1),1);
    FNR = zeros(size(data,1),1);
    
    xdepth = length(data(1,:));
    covdepth = xdepth+1;
    
%     if (sum(results1 <= energycutoff) < 20)
%         disp(strcat({'Too few datapoints for cutoff '},num2str(energycutoff),'; skipping.'));
%         continue;
%     end

    confidence_passes = [];
    
    % Passed to contour3d - added to each iteration
    indexout = [];
    mfout = [];
    mfout_precision = [];
    indexout_precision = [];

    TPR_binned_total = zeros(binCount);
    FPR_binned_total = zeros(binCount);
    
    for l = 1:loopn
        
        hyp = [];

        % Train hyperparameters
        hyp.mean = zeros(1,1);
        hyp.cov(1:covdepth) = log(1);
        hyp.lik(1:1) = -.2;
        
        trainData = data;
        trainy = (results1 <= energycutoff);
        
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
        testIndices = randomIndices(trainData,length(testx));
        % Unlike most of the other experiments, we want predictions for
        % both test and training points.
        testx = trainData(testIndices,:);
        train2x = trainData(train2Indices,:);
        trainData([testIndices; train2Indices],:) = [];
        trainy([testIndices; train2Indices]) = [];

        trainy_total = [trainy_total; trainy];

        dbstop if error
        % Minimize hyperparameters
        hyp = minimize(hyp, @gp, -1000, inffunc, meanfunc, covfunc, likfunc, trainData, trainy);
   
        % Run GP
        [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainData,trainy,train2x);
        
        %mf = normalizeer(mf);
        %mf = +(mf >= gpCutoff);
        
%         TPR_binned = zeros(binCount,binCount);
%         FPR_binned = zeros(binCount,binCount);
%         TNR_binned = zeros(binCount,binCount);
%         FNR_binned = zeros(binCount,binCount);
        
        train2y = zeros(size(train2x,1),1);
        for i = 1:size(train2y,1)
            %bin_index = ceil(((train2x(i,:)-PCA_min) ./ (PCA_max-PCA_min)) * binCount);
            % TPR=3, FPR=2, TNR=0, FNR=1
            train2y(i) = (mf(i) >= gpCutoff) * 2 + ((groundtruth(train2Indices(i)) >= physicalCutoff) == 1);
            %FPR_binned(bin_index(1),bin_index(2)) = FPR_binned(bin_index(1),bin_index(2)) + (mf(i) == 1 & (groundtruth(i) >= cutoff) == 0);
            
            %TNR_binned(bin_index(1),bin_index(2)) = TNR_binned(bin_index(1),bin_index(2)) + (mf(i) == 0 & (groundtruth(i) >= cutoff) == 0);
            %FNR_binned(bin_index(1),bin_index(2)) = FNR_binned(bin_index(1),bin_index(2)) + (mf(i) == 0 & (groundtruth(i) >= cutoff) == 1);
        end
        
        % Don't split on results, but include them for easy bin confidence
        % computing
        tree = kdCreateTree([train2x train2y],binThreshold,size(train2x,2));
        bins = kdGetAllBins(tree);
        fprintf('Number of bins: %d\n',size(bins,1));
        bin_size_sum = 0;
        for i = 1:size(bins,1)
            bin_size_sum = bin_size_sum + size(bins{i,1},1);
        end
        fprintf('Average bin size: %d\n',bin_size_sum / size(bins,1));
        
        for i = 1:size(bins,1)
            % This is one more column than we are normally retrieving (the
            % train2y column has been appended), but kd tree ignores it
            % anyway, so should not impact testing consistency
            testbin = kdGetBin(tree,bins{i,1}(1,:));
            % Copied verbatim from binning code below
            bin_TP = sum(testbin(:,size(data,2)+1) == 3);
            bin_FP = sum(testbin(:,size(data,2)+1) == 2);
            confidence = bin_TP / (bin_TP + bin_FP);

            if(isnan(confidence) || confidence < confidence_cutoff)
                fprintf('Bin %d (depth %d): filtered\n',i,bins{i,2});
                %eliminate(i) = 1;
            else
                fprintf('Bin %d (depth %d): kept\n',i,bins{i,2});
            end
            
            disp(bins{i,1});
        end
        
        %confidence = TPR_binned ./ (TPR_binned + FPR_binned);
        %passing_bins = (confidence >= confidence_cutoff & binCounts >= support_threshold);
        %confidence_passes(end+1) = sum(passing_bins(:));

        % Filter out test points outside areas of confidence
        if(filter_with_confidence)
            eliminate = zeros(size(testx,1),1);
            for i = 1:size(testx,1)
                % Insert a garbage value at the end of testx vector, since
                % the input must be the same size as a test point plus the
                % TP/FP/TN/FN flag
                testbin = kdGetBin(tree,[testx(i,:) 0]);
                
                % True positive rate confidence
                bin_TP = sum(testbin(:,size(data,2)+1) == 3);
                bin_FP = sum(testbin(:,size(data,2)+1) == 2);
                confidence = bin_TP / (bin_TP + bin_FP);
                % True negative rate confidence
%                 bin_TP = sum(testbin(:,3) == 0);
%                 bin_FN = sum(testbin(:,3) == 1);
%                 confidence = bin_TN / (bin_TN + bin_FN);
                
                %assert(((isnan(confidence) || confidence < confidence_cutoff) && passing_bins(bin_index(1),bin_index(2))==0) || (confidence >= confidence_cutoff && passing_bins(bin_index(1),bin_index(2))==1));
                if(isnan(confidence) || confidence < confidence_cutoff)
                    eliminate(i) = 1;
                end
            end
            
            testIndices_final = testIndices(~logical(eliminate));
            testx = testx(~logical(eliminate),:);
            %disp(strcat('Eliminated ','',num2str(sum(eliminate)),' elements from final test set.'));
        else
            testIndices_final = testIndices;
        end
        
        if size(testx,1) < 10
            disp('Not enough valid test points; skipping iteration.');
            continue;
        end
                
        % Keeps track of which predictions were from which nodes
        test_containing_nodes = zeros(size(testIndices,1),1);
        % We want these for all test indices including the eliminated ones
        for i = 1:size(test_containing_nodes,1)
            idx = kdGetBinIndex(tree,[data(testIndices(i),:) 0]);
            test_containing_nodes(i) = idx;
        end
        test_containing_nodes_filtered = test_containing_nodes(~eliminate);
        
        % Run GP
        [mf, s2f, fmu, fs2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc,trainData,trainy,testx);
        
        %mf = normalizeer(mf);
        %mf = +(mf >= cutoff);
        res = +(groundtruth(testIndices_final) >= cutoff);
        
        nodeIndices = kdGetAllNodeIndices(tree);
        
        %% Best node calculation
        node_precisions = zeros(size(nodeIndices));
        % Generate the precision of all bins/nodes
        for i = 1:size(nodeIndices,1)
            [node_precisions(i), ~] = precisionAndRecall({mf(test_containing_nodes_filtered==nodeIndices(i)) >= gpCutoff},{res(test_containing_nodes_filtered==nodeIndices(i))},true);
        end
        mf_precision = zeros(size(test_containing_nodes,1),1);
        for i = 1:size(mf_precision)
            mf_precision(i) = node_precisions(nodeIndices==test_containing_nodes(i)) > contourPrecisionCutoff;
        end
        %% End best node calculations
        
        mfout = [mfout; mf];
        mfout_precision = [mfout_precision; mf_precision];
        indexout = [indexout; testIndices_final];
        indexout_precision = [indexout_precision; testIndices];
        
        % Generate contour based on GP outputs
%         [mf_cont, s2f_cont, fmu_cont, fs2_cont] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, trainData, trainy, contour_inputs);
%         mf_cont = normalizeer(mf_cont) .* ~eliminate_contour;
%         contour_values_iter = reshape(mf_cont,contour_granularity,contour_granularity);
%         contour_values = contour_values + (contour_values_iter > .6);
        
        dbstop if error;    
        
        % Generate ROC curve
        try
            [perfx,perfy,perft,auc(end+1)] = perfcurve(res',normalizeer(mf)',1);
            %rocResults = [rocResults; [perfx perfy]];
            [rocx(:,end+1),rocy(:,end+1)] = rotandproject(perfx,perfy,.01);
        catch err
            auc(l) = NaN;
        end

        disp(strcat('Completed iteration: ',num2str(l)));
        [precision, recall] = precisionAndRecall({mf >= gpCutoff},{res},true);
        disp(strcat('Prediction points:',num2str(size(testIndices_final,1))));
        disp('  TP     FP     TN     FN');
        [tp, fp, tn, fn] = getTestResults(mf >= gpCutoff,res);
        disp([tp, fp, tn, fn]);
        disp('Precision / Recall');
        disp([precision recall]);
        predictions{l} = (mf >= gpCutoff);
        trueValues{l} = res;
        
        %passing_bin_stats = zeros(confidence_passes(end),4);
        % Concatenate all high-confidence bins into an Nx4 vector showing
        % TPR/FPR/TNR/FNR
%         for i = 1:numel(passing_bins)
%             if(passing_bins(i))
%                 passing_bin_stats(k,:) = [TPR_binned(i) FPR_binned(i) TNR_binned(i) FNR_binned(i)];
%                 k = k+1;
%             end
%         end
%         [Y,I] = sort(passing_bin_stats(:,1),'descend');
%         passing_bin_stats_sorted = passing_bin_stats(I,:);
%         disp('Passing bin stats (TPR/FPR/TNR/FNR)');
%         disp(passing_bin_stats_sorted);
    end
    
%    fprintf('Average number of confident bins (out of %d): %f\n',binCount*binCount,mean(confidence_passes));

%     h = figure(energycutoff+1);
%     TPR_hist_data = zeros(sum(TPR_binned_total(:)),2);
%     s = 1;
%     for xi = 1:binCount
%         for yi = 1:binCount
%             for k = 1:TPR_binned_total(xi,yi)
%                 TPR_hist_data(s,:) = [xi yi];
%                 s = s + 1;
%             end
%         end
%     end
%     hold on
%     hist3(TPR_hist_data,[binCount binCount]);
%     title(sprintf('TPR for GraspIt GP: cutoff=%d',energycutoff));
%     xlabel('PCA1');
%     ylabel('PCA2');
%     zlabel(sprintf('Total true positives in bin'));
%     hold off
    
    
    [ave,uppererr,lowererr,upperstd,lowerstd,kstat,kstatsterr,kstatstd] = rotandextrap(rocx,rocy,size(auc,1));
    
    if(filter_with_confidence)
        contour_file = strcat(savedir,'hybrid_contours\',sprintf('contour_kdhybrid_energy_%d_conf_%d_bin_%d_%dD',energycutoff,confidence_cutoff*10,binThreshold,finalDimensionNum));
    else
        contour_file = strcat(savedir,'hybrid_contours\',sprintf('contour_kdhybrid_energy_%d_nofilter_%dD',energycutoff,finalDimensionNum));
    end
    generateContour(data_pca,groundtruth,indexout_precision,mfout_precision,contour_file);  
    
    fig = figure;
    set(fig, 'units', 'inches', 'pos', [8 5 3.25 3])
    set(gca,'FontSize',10)
    fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
    %fill(([uppererr(:,1)' rot90(lowererr(:,1),2)'].*100),([uppererr(:,2)' rot90(lowererr(:,2),2)'].*100),[.75 .75 .75])
    hold on
    plot(ave(:,1).*100,ave(:,2).*100,'k','linewidth',2)
    %plot(rocResults(:,1).*100,rocResults(:,2).*100,'k','linewidth',2)
    plot([0 100],[0 100],'color',[.5 .5 .5])
    set(gca,'box','on','position',[.17,.15,.78,.8])
    axis square
    axis([0 100 0 100])
    ylabel({'TPR';'(%)'})
    xlabel('FPR (%)')
    set(get(gca,'YLabel'),'Rotation',0)
    plot([40 45],[52,60],'k','linewidth',1)
    set(get(gca,'YLabel'),'Position',[-13 45. 1.001])
    text(10,84,'GP with PC1','FontSize',10)
    text(10,8,'Random Guess','FontSize',10)

    disp('Overall results (hybrid)');
    if(filter_with_confidence)
        disp(strcat('Confidence threshold=',num2str(confidence_cutoff)));
    end
    if(~filter_with_confidence)
        disp('All datapoints used');
    else
        disp('Filtered datapoints');
    end
    
    %Display AUC Info
    disp('AUC ave err std')
    disp(mean(auc))
    disp(std(auc)/(size(auc,1))^.5)
    disp(std(auc))
    
    
    [precision_macro, recall_macro] = precisionAndRecall(predictions,trueValues,true);
    disp('Precision / Recall macro');
    disp([precision_macro recall_macro]);
    disp('');
    [precision_micro, recall_micro] = precisionAndRecall(predictions,trueValues,false);
    disp('Precision / Recall micro');
    disp([precision_micro recall_micro]);

end

disp('    TPR       FPR       TNR       FNR       cutoff');
disp(summaryStats);

thesisCleanup;
end