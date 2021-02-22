function M = unfold(T, mode)
% unfold function

if ( mode == 1 )
% fibers in mode 1, taken in order by frontal slices
  M = squeeze( T(:,:,1) ) ; 
  for i = 2 : size(T,3)
    M = [M squeeze(T(:,:,i)) ] ;
  end
elseif ( mode == 2 )
  M = T(:,:,1)' ;
  for i = 2 : size(T,3)
    M = [ M T(:,:,i)' ];
  end
elseif ( mode == 3 )
  M = squeeze(T(:,1,:))' ;
  for i = 2 : size(T,2)
    M = [ M squeeze(T(:,i,:))' ] ;
  end
end
  
endfunction
  