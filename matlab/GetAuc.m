function [auc, fpr, tpr] = GetAuc(y, f)
    % Input:
    % x - matrix [objects * features]
    % y - labels. y \in {0, 1} ({-1, 1} is also supported)
    % f - callable functions
    % Output:
    %    FIXIT
    % Example:
    %   f = rand(10, 1);
    %   y = round(rand(10, 1));
    %   [auc, fpr, tpr] = GetAuc(y, f);
    %   plot(fpr, tpr)
    nObjects = size(y, 1);
    
    nPositive = sum(y == 1);
    nNegative = nObjects - nPositive;
    
    [B, I] = sort(f, 'descend');
    % sort y
    y = y(I);

    fpr = [0]; % Faulse positive rate
    tpr = [0]; % true positive rate
    auc = 0; % area under curve
    
    for i = 1:nObjects
        if y(i) == 1
            fpr = [fpr; fpr(end)];
            tpr = [tpr; tpr(end) + 1 / nPositive];
        else
            fpr = [fpr; fpr(end) + 1 / nNegative];
            tpr = [tpr; tpr(end)];
            auc = auc + tpr(end) / nNegative;
        end
    end
end
