%% Handwritten digit classification using HOSVD

%clear ; close all; clc

%% ==================== Part 1: Setting Up The Data ============================

load('mnist_dataset.mat'); % training data stored in arrays X, y
fprintf('Data loaded.\n');
% matrix X is of dimensions 5000x400 and contains the images of digits
% vector y is of dimensions 5000x1 and contains images labels
% 5000 images of digits of 20x20 pixels
% each class has precisely 500 digits


% in this dataset all digits are ordered, so we need to shuffle them, but
% then we need to shuffle the vector of labels - y correspondingly
% we need to shuffle elemenats from 1 to 5000
shuffle = randperm(5000);
X = X(shuffle, :);
y = y(shuffle);
fprintf('Data shuffled.\n');

% number 10 marks digit 0, so we can change that too in vector y
for i = 1 : 5000
  if(y(i) == 10)
    y(i) = 0;
  endif
endfor

%imagesc(reshape(X(15,:), 20, 20))


% now we need to put the images into tensor
[n1 n2] = size(X);

Timg = zeros(sqrt(n2), sqrt(n2), n1);
for i = 1:n1
  Timg(:,:,i) = reshape(X(i,:), sqrt(n2), sqrt(n2));
end

% splitting into train and test datasets, both digits and corresponding labels
trainimage = Timg(:,:,1:4000);
trainlabel = y(1:4000);
testimage = Timg(:,:,4001:5000);
testlabel = y(4001:5000);
fprintf('Data split into training and test.\n');

%imagesc(trainimage(:,:,115))
%imagesc(reshape(X(115,:), 20, 20))
% everything is ok

% distribution of digits is about the same

%% ==================== Part 2: Algorithm 1 ====================================
fprintf('Algorithm 1...\n');


% 1.) we need to create 10 tensors which will contain digits from the same class
ndigits = zeros(1,10);
for i = 1:10
  ndigits(i) = sum(trainlabel == i-1);
end

fprintf('Splitting data into same digit tensors...\n');
% here we have put tensors in cell arrays which can contain variables of 
% different sizes
for i = 1:10
  M{i} = samedigittensor(trainimage, trainlabel, n2, ndigits, i-1);
endfor
fprintf('Data split.\n');


% 2.) HOSVD + 3.) basis matrices
% we could calculate third parameter k which gives best results, i.e. which
% is the best approximation
% results vary a lot for different k, for k = 20 we got good results

fprintf('Calculating basis matrices for each class...\n');
k = 20;
for i = 1:10
  B{i} = getbasismat(M, i, k);
  % here we can adjust parameter k
  
endfor
fprintf('Basis matrices calculated.\n');

basisall = zeros(sqrt(n2), sqrt(n2), 10 * k);
for i = 1:10
  basisall(:,:,(k * i) - (k-1) : k * i) = B{i};
endfor


% 3.) Algorithm 1 - precision
%digitclassify(testimage(:,:,244), basisall ,k)
%imagesc(testimage(:,:,244))

% testing the precision of algorithm 1
fprintf('Calculating Algorithm 1 precision...\n');
brojac = 0;
for i = 1 : 1000
  citaj = digitclassify(testimage(:,:,i), basisall , k);
  if(citaj == testlabel(i)) 
    brojac = brojac + 1;
  endif
endfor

fprintf('Algorithm 1 precision: %d %%\n', brojac/10); 

 
%% ==================== Part 3: Algorithm 2 ====================================
fprintf('Algorithm 2...\n');

% 1.) creating tensor D

% mode digit will be of size min(ndigits)

D = zeros(n2, min(ndigits), 10);
for i = 1 : 10
  for j = 1 : min(ndigits)
    D(:,j,i) = vec(M{i}(:,:,j));
  endfor
endfor

% 2.) HOSVD tenzora D
%[DS, DU1, DU2, DU3] = HOSVD(D);
% problem with calculating HOSVD ---> out of memory or dimension...too big?
% problem is the line [ DU3, ~, ~ ] = svd( unfold( D, 3) ) ;

[ DU1, ~, ~ ] = svd( unfold( D, 1) ) ;
[ DU2, ~, ~ ] = svd( unfold( D, 2) ) ;

p = 64;
q = 64;

% 3.)
F = tm_multiply( tm_multiply(D, DU1(:, 1:p)', 1) , DU2(:, 1:q)', 2);

%===========================================
% 3.) Calculating reduced tensor F

% first we can plot the values of singular values so we could determine which
% are potentially good candidates for parameters p and q

%Pixel_mode_singular_values = zeros(100);
%for i = 1 : 100
  %Pixel_mode_singular_values(i) = frobnorm(DS(i,:,:));
%endfor

%Digit_mode_singular_values = zeros(100);
%for i = 1 : 100
  %Digit_mode_singular_values(i) = frobnorm(DS(:,i,:));
%endfor

%plot(1:100, Pixel_mode_singular_values, "ko");

% this piece of code cannot be run because we couldn't calculate the complete
% HOSVD of tensor D becuase of its size
%=============================================

% 4.)

fprintf('Calculating basis vectors...\n');
k_column_base = 20;
for i = 1 : 10
  [U, ~, ~] = svd(F(:,:,i));
  B2{i} = U(:, 1:k_column_base);
endfor
fprintf('Basis vectors calculated.\n');
 
% 5.) Algorithm 2 - precision

fprintf('Calculating Algorithm 2 precision...\n');
brojac2 = 0;
for i = 1 : 1000
  d_p =  DU1(:, 1:p)'   *   X(4000 + i,:)';
  
  citaj = digitclassify2(B2, d_p);
  
  if(citaj == testlabel(i)) 
    brojac2 = brojac2 + 1;
  endif
endfor

fprintf('Algorithm 2 precision: %d %%\n', brojac2/10);

%============================= Part 4: Plots ===================================

% Plot 1.) for Algorithm 1
%===============================================================
% first we plot the error rates per different k, for algorithm 1
% it runs for about 15 min

%different_k_error = zeros(1,15);
%for different_k = 6:20
  %for i = 1:10
  %B{i} = getbasismat(M, i, different_k);
  %endfor

  %basisall = zeros(sqrt(n2), sqrt(n2), 10 * different_k);
  
  %for i = 1:10
  %basisall(:,:,(different_k * i) - (different_k - 1) : different_k * i) = B{i};
  %endfor
  
  %brojac = 0;
  %for i = 1 : 1000
    %citaj = digitclassify(testimage(:,:,i), basisall , different_k);
    %if(citaj == testlabel(i)) 
      %brojac = brojac + 1;
    %endif
  %endfor
  %different_k_error(different_k - 5) = (1000 - brojac) / 1000;
%endfor

%plot(6:20, different_k_error, "-ko")
%xlabel ("Number of basis matrices");
%ylabel ("Error rate");
%==================================================================




% Plot 2.) for Algorithm 2
%===============================================================
% it runs for about 3 min

%different_p = [32 64 64 32 48 64];
%different_q = [32 32 48 64 64 64];

% we will test for 6 combinations of p and q, each with 16 different k basis vectors
%error_rate_matrix = zeros(6, 15);

%for i = 1:6
  %for different_k_cb = 6:20

    %F = tm_multiply( tm_multiply(D, DU1(:, 1:different_p(i))', 1) , DU2(:, 1:different_q(i))', 2);
    %for j = 1 : 10
      %[U, ~, ~] = svd(F(:,:,j));
      %B2{j} = U(:, 1:different_k_cb);
    %endfor
    
    %brojac2 = 0;
    %for j = 1 : 1000
      %d_p =  DU1(:, 1:different_p(i))'   *   X(4000 + j,:)';
      %citaj = digitclassify2(B2, d_p);
      %if(citaj == testlabel(j)) 
        %brojac2 = brojac2 + 1;
      %endif
    %endfor
    
    %error_rate_matrix(i, different_k_cb - 5) = (1000 - brojac2) / 1000;
    
  %endfor
%endfor


%figure; plot(6:20, error_rate_matrix(1,:), '+-');
%xlabel("Number of basis vectors"),
%ylabel("Error rate")
%hold on; plot(6:20, error_rate_matrix(2,:), 'o-r'); 
%plot(6:20, error_rate_matrix(3,:), '*-g');
%hold off;
%legend('32x32', '64x32', '64x48')


%figure; plot(6:20, error_rate_matrix(4,:), '+-');
%xlabel("Number of basis vectors"),
%ylabel("Error rate")
%hold on; plot(6:20, error_rate_matrix(5,:), 'o-r'); 
%plot(6:20, error_rate_matrix(6,:), '*-g');
%hold off;
%legend('32x64', '48x64', '64x64')

%==================================================================


%========================== Part 5: direct comparison ==========================
% here we can see directly to which class did each algorithm assign each digit
% from the test dataset and what it really is

for i = 1 : 1000
  x = input("Continue?: ");
  % any number other than 0 for yes, 0 for no
  if (x == 0)
    break;
  endif
  
  imagesc(testimage(:,:,i))
  fprintf('Algorithm 1 predict: %d \n', digitclassify(testimage(:,:,i), basisall, k))
  
  d_p =  DU1(:, 1:p)'   *   X(4000 + i,:)';
  fprintf('Algorithm 2 predict: %d \n', digitclassify2(B2, d_p))
  
  fprintf('Number on the picture: %d \n', testlabel(i))
endfor
