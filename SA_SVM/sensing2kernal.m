%% Aware Sensing 2 kernal
% Referencing sensing 2 Kernel from:
% http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6854140&tag=1
% Bowen Song 
% Apr 5th, 2018

function K2 = sensing2kernal(x_hat_i,x_hat_j,alpha)


K2 = gammaln(x_hat_i+x_hat_j+1)-gammaln(x_hat_i+1)-gammaln(x_hat_j+1)+alpha;


end