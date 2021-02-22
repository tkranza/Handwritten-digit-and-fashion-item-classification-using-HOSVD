function R = tm_multiply( T, A, mode )
% function which calculates tensor-matrix multiplication in given mode

if ( size(A,2) ~= size(T,mode) )
    error('>> number of columns of matrix is different from corresponding tensor dimension!')
end
dR = size(T) ; dR(mode) = size(A,1) ; 

R = fold( A * unfold(T, mode), mode, dR ) ;

endfunction
