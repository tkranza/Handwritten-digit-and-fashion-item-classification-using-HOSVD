function digit = digitclassify(D, basisall, k)
% function which return classification of unknown digit D / Algorithm 1

D_norm = D/frobnorm(D);

R = zeros(1, 10);

for i = 1:10
  for j = 1:k
    
    R(i) = R(i) + sum(sum(D_norm .* basisall(:,:,(i-1) * k + j)))^2;
    
  end
  
  R(i) = 1 - R(i);
  
end

for i = 1:10
  if(R(i) == min(R))
    digit = i-1;
  end
end

endfunction