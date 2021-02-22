function digit = digitclassify2(B2, d_p)
% % function which return classification of unknown digit d_p / Algorithm 2

R = zeros(1, 10);

for i = 1:10
  R(i) = frobnorm(d_p - B2{i} * B2{i}' * d_p);
end


for i = 1:10
  if(R(i) == min(R))
    digit = i-1;
  end
end

endfunction