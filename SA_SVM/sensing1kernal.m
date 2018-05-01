%% Aware Sensing 1 kernal
% Referencing sensing 2 Kernel from:
% http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6854140&tag=1
% Bowen Song
% Apr 5th, 2018

function K1 = sensing1kernal(x_hat_i,x_hat_j)

K1 = zeros(size(x_hat_i,1),size(x_hat_j,1));
global SA_n
[n,m] = size(x_hat_i);

% theones = ones(1,m);
parfor i = 1:size(x_hat_i,1)
    tic
%     parfor j = 1:size(x_hat_j,1)
%         K1(i,j) = sum(gammaln(SA_n.*(x_hat_i(i,:)+x_hat_j(j,:))+theones)...
%             -gammaln(SA_n.*x_hat_i(i,:)+theones)-gammaln(SA_n.*x_hat_j(j,:)+theones),2);
%     end
K1(i,:) = sum(gammaln(SA_n.*x_hat_i(i,:)+SA_n.*x_hat_j+1)-gammaln(SA_n.*x_hat_i(i,:)+1)-gammaln(SA_n.*x_hat_j+1),2);
    fprintf("The %d loop time outof %d loops ",i,n)
    toc
end

%K1(i,:) = sum(gammaln(SA_n.*x_hat_i(i,:)+SA_n.*x_hat_j+1)-gammaln(SA_n.*x_hat_i(i,:)+1)-gammaln(SA_n.*x_hat_j+1),2);


end