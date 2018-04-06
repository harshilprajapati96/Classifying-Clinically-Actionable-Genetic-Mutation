%% Aware Sensing 2 kernal
% Referencing sensing 2 Kernel from:
% http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6854140&tag=1
% Bowen Song 
% Apr 5th, 2018

function K2 = sensing2kernal(x_hat_i,x_hat_j,X_train_doc_w,vocab_len)


x_i_w = X_train_doc_w(x_hat_i,:);
x_j_w = X_train_doc_w(x_hat_j,:);
N_i = sum(x_i_w,2);
N_j = sum(x_j_w,2);
tempnum = factorial(x_i_w+x_j_w)*factorial(N_i)*factorial(N_j);
tempde = factorial(x_i_w)*factorial(x_j_w)*factorial(N_i+N_j+vocab_len-1);

K2 = log(prod(tempnum/tempde,2));







end