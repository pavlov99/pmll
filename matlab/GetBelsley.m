function [conditionalityIndexes, Q] = GetBelsley(x)
    % Caclulate Belsley matrix
    % Input: x - matrix [objects * features]
    % Output: Q - FIXIT what is it?
    [U,S,V] = svd(x, 'econ');
    
    % divide each row of V.^2 by S(i).^2 - singular value
    Q = diag(1./diag(S).^2) * V.^2;
    
    % Normalize Q: column total is 1  
    Q = Q * diag(1 ./ sum(Q));
    
    conditionalityIndexes = max(diag(S)) ./ diag(S);
end