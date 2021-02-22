function [ S, U1, U2, U3 ] = HOSVD( A )
% Calculates HOSVD of tensor A
 
[ U1, ~, ~ ] = svd( unfold( A, 1) ) ;
[ U2, ~, ~ ] = svd( unfold( A, 2) ) ;
[ U3, ~, ~ ] = svd( unfold( A, 3) ) ;
S = tm_multiply(tm_multiply(tm_multiply(A,U1',1),U2',2),U3',3) ; 

end