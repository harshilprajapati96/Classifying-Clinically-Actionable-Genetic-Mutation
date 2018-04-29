%% Aware Sensing 1 kernal
% Referencing sensing 2 Kernel from:
% http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6854140&tag=1
% Bowen Song
% Apr 5th, 2018

function K1 = sensing1kernal(x_hat_i,x_hat_j)

K1 = zeros(size(x_hat_i,1),size(x_hat_j,1));
global SA_n

for i = 1:size(x_hat_i,1)
K1(i,:) = sum(gammaln(SA_n.*x_hat_i(i,:)+SA_n.*x_hat_j+1)-gammaln(SA_n.*x_hat_i(i,:)+1)-gammaln(SA_n.*x_hat_j+1),2);

end

end