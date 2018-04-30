%% Aware Sensing 1 kernal
% Referencing sensing 2 Kernel from:
% http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6854140&tag=1
% Bowen Song
% Apr 5th, 2018

function K1 = sensing1kernal(x_hat_i,x_hat_j)

K1 = zeros(size(x_hat_i,1),size(x_hat_j,1));
global SA_n
[n,m] = size(x_hat_i);

theones = ones(1,m);
for i = 1:size(x_hat_i,1)

    parfor j = 1:size(x_hat_j,1)
        
        
        K1(i,j) = sum(gammaln(SA_n.*(x_hat_i(i,:)+x_hat_j(j,:))+theones)...
            -gammaln(SA_n.*x_hat_i(i,:)+theones)-gammaln(SA_n.*x_hat_j(j,:)+theones),2);
        
    end

end

% for i = 1:size(x_hat_i,1)
%     tic
%     parta = repmat(SA_n.*x_hat_i(i,:),n,1);
%     partab = parta+theones;
%     a = gammaln(SA_n.*x_hat_j+partab);
%     b = gammaln(partab);
%     c = gammaln(SA_n.*x_hat_j+theones);
%     temp = sum(a-b-c,2);
%     toc
%     tic
%K1(i,:) = sum(gammaln(SA_n.*x_hat_i(i,:)+SA_n.*x_hat_j+1)-gammaln(SA_n.*x_hat_i(i,:)+1)-gammaln(SA_n.*x_hat_j+1),2);
% toc
% end

end