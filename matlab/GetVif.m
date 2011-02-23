function [vif] = GetVif(x)
    % Calculate variance inflation factor 
    % Input: x - matrix [objects * features]
    % Output: vif - vector [features * 1]
    
    [nObjects, nFeatures] = size(x);
    vif = zeros(nFeatures, 1); 
    
    for iFeature = 1:nFeatures
        vif(iFeature) = sum((x(:, iFeature) - mean(x(:, iFeature))).^2) / ...
        sum(GetRegressionResiduals(x(:, [1:iFeature - 1,iFeature + 1:end]), ...
            x(:, iFeature)).^2);
    end
end

function [residuals] = GetRegressionResiduals(x, y)
    residuals = y - x * (x' * x)^(-1) * x' * y;
end