%% Fashion item classification with Algorithm 2

%clear ; close all; clc

%% ==================== Part 1: Setting Up The Data ============================

load('fashion-mnist.mat') % we load images to the variable FX, and labels to Fy

[Fn1 Fn2] = size(FX);

% tensor creation
FTimg = zeros(sqrt(Fn2), sqrt(Fn2), Fn1);
for i = 1:Fn1
  FTimg(:,:,i) = reshape(FX(i,:), sqrt(Fn2), sqrt(Fn2))';
end

% splitting to train and test datasets, items and corresponding labels
Ftrainimage = FTimg(:,:,1:4000);
Ftrainlabel = Fy(1:4000);
Ftestimage = FTimg(:,:,4001:5000);
Ftestlabel = Fy(4001:5000);
fprintf('Data split into training and test.\n');


fashion = {"T-shirt/top"; "Trouser"; "Pullover"; "Dress"; "Coat"; "Sandal";
               "Shirt"; "Sneaker"; "Bag"; "Ankle boot" };


                 
% we can check how does data look       
for i = 1 : 4000
  x = input("Continue?: ");
  if (x == 0)
    break;
  endif
  
  imagesc(Ftrainimage(:,:,i))
  colormap(gray)
  
  fprintf('Item on the picture: %s \n', fashion{Ftrainlabel(i)+1})
endfor

%============================== Algorithm 2 ====================================


fprintf('Algorithm 2...\n');

% 1.) formation of tensor FD

Fndigits = zeros(1,10);
for i = 1:10
  Fndigits(i) = sum(Ftrainlabel == i-1);
end

for i = 1:10
  FM{i} = samedigittensor(Ftrainimage, Ftrainlabel, Fn2, Fndigits, i-1);
endfor
fprintf('Same item tensor created.\n');

FD = zeros(Fn2, min(Fndigits), 10);
for i = 1 : 10
  for j = 1 : min(Fndigits)
    FD(:,j,i) = vec(FM{i}(:,:,j));
  endfor
endfor

% 2.) HOSVD of tensor FD

[ FDU1, ~, ~ ] = svd( unfold( FD, 1) ) ;
[ FDU2, ~, ~ ] = svd( unfold( FD, 2) ) ;

Fp = 64;
Fq = 64;

% 3.)

FF = tm_multiply( tm_multiply(FD, FDU1(:, 1:Fp)', 1) , FDU2(:, 1:Fq)', 2);

% 4.)

fprintf('Calculating basis vectors...\n');
Fk_column_base = 28;
for i = 1 : 10
  [FU, ~, ~] = svd(FF(:,:,i));
  FB2{i} = FU(:, 1:Fk_column_base);
endfor
fprintf('Basis vectors calculated.\n');
 
% 5.) Algorithm 2 - precision

fprintf('Calculating Algorithm 2 precision...\n');
Fbrojac2 = 0;
for i = 1 : 1000
  Fd_p =  FDU1(:, 1:Fp)'   *   vec(Ftestimage(:,:,i));
  
  citaj = digitclassify2(FB2, Fd_p);
  
  if(citaj == Ftestlabel(i)) 
    Fbrojac2 = Fbrojac2 + 1;
  endif
endfor

fprintf('Algorithm 2 precision: %d %%\n', Fbrojac2/10);

% about 75% precision

%======================================================================


for i = 1 : 1000
  x = input("Continue?: ");
  if (x == 0)
    break;
  endif
  
  imagesc(Ftestimage(:,:,i))
  colormap(gray)
  
  Fd_p =  FDU1(:, 1:Fp)'   *   vec(Ftestimage(:,:,i));
  fprintf('Algorithm 2 predict: %s \n', fashion{digitclassify2(FB2, Fd_p) + 1})
  
  fprintf('Item on the picture: %s \n', fashion{Ftestlabel(i)+1})
endfor

% it mixes sneaker and sandal a bit
