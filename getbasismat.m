function R = getbasismat(T, i, k)
 % function which takes in tensor T with the digits from the same class
 % and returns k basis matrices
 % parameter k is the number of basis matrices in each class
  
[S, U1, U2, ~] = HOSVD(T{i});

R = zeros(20,20,k);

for j=1:k
  R(:,:,j) = U1 * S(:,:,j) * U2';
  
  % normalization
  R(:,:,j) = R(:,:,j)/frobnorm(R(:,:,j));
end

endfunction