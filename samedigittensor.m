function T = samedigittensor(trainimage, trainlabel, n2, ndigits, k)
% function which returns tensor containing the digits from the same class

T = zeros(sqrt(n2), sqrt(n2), ndigits(k+1));
idx = find(trainlabel == k);
T = trainimage(:,:,idx);

endfunction