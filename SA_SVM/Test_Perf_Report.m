%% %% SVM Classifier for Text Document
% MATLAB R2017b
% Bowen Song U04079758
% casenumber = input('Enter a number: ');
%
% switch casenumber
%     case 1
disp('linear News 20 group OVA')
%% News 20 group OVA
clear all
Preprocessing_new20;
disp("Preprocessing is done:")
boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "USnews_linear_OVA";
tic
function_linear_OVA(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,filename)
toc
%     case 2
disp('linear News 20 group OVO')
% %% News 20 group OVO
% % preprocessing
% clear all
% Preprocessing_new20;
% disp("Preprocessing is done:")
% boxcon_power = -7:13;
% boxcon = 2.^boxcon_power;
% filename = "USnews_linear_OVO";
% function_linear_OVO(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,filename)
% %     case 3
% disp('linear Clinic OVA')
%% Clinc result
% preprocessing
clear all
tic
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Data_Cancer_Filtered_LeastFreq_Words.mat');
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Label_Cancer.mat');
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Data_Cancer_Filtered_LeastFreq_Words.mat');
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Label_Cancer');
vocab = importdata('../Cancer_Detection_Data_Decreased_Vocab_Size/vocabulary_Filtered_LeastFreq_Words.txt');
toc
boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "Clinc_linear_OVA";
function_linear_OVA(train_data_cancer,test_data_cancer,Train_Label_Cancer,...
    Test_Label_Cancer,vocab,boxcon,filename)
%     case 4
disp('linear Clinic OVO')
% %% Clinc result
% % preprocessing
% clear all
% tic
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Data_Cancer_Filtered_LeastFreq_Words.mat');
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Label_Cancer.mat');
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Data_Cancer_Filtered_LeastFreq_Words.mat');
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Label_Cancer');
% vocab = importdata('../Cancer_Detection_Data_Decreased_Vocab_Size/vocabulary_Filtered_LeastFreq_Words.txt');
% toc
% boxcon_power = -7:13;
% boxcon = 2.^boxcon_power;
% filename = "Clinc_linear_OVO";
% function_linear_OVO(train_data_cancer,test_data_cancer,Train_Label_Cancer,...
%     Test_Label_Cancer,vocab,boxcon,filename)


%     case 5
disp('News 20 group rbf OVA')
%% News 20 group OVA
% preprocessing
% tic
clear all
Preprocessing_new20;
disp("Preprocessing is done:")
% toc
boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
rbf_power = -7:9;
rbf_sig = 2.^rbf_power;
filename = "USnews_rbf_OVA";
addpath('libsvm-320/matlab');
function_rbf_OVA(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,rbf_sig,filename)
%     case 6
% disp('News 20 group rbf OVO')
% %% News 20 group OVO
% % preprocessing
% clear all
% Preprocessing_new20;
% disp("Preprocessing is done:")
% rbf_power = -7:9;
% rbf_sig = 2.^rbf_power;
% boxcon_power = -7:13;
% boxcon = 2.^boxcon_power;
% filename = "USnews_rbf_OVO";
% function_rbf_OVO(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,rbf_sig,filename)
%     case 7
disp('Clinic rbf OVA')
%% Clinc result
% preprocessing
clear all
tic
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Data_Cancer_Filtered_LeastFreq_Words.mat');
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Label_Cancer.mat');
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Data_Cancer_Filtered_LeastFreq_Words.mat');
load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Label_Cancer');
vocab = importdata('../Cancer_Detection_Data_Decreased_Vocab_Size/vocabulary_Filtered_LeastFreq_Words.txt');
toc
rbf_power = -7:9;
rbf_sig = 2.^rbf_power;
boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "Clinc_rbf_OVA";
addpath('libsvm-320/matlab');
function_rbf_OVA(train_data_cancer,test_data_cancer,Train_Label_Cancer,...
    Test_Label_Cancer,vocab,boxcon,rbf_sig,filename)
%     case 8
% disp('Clinic rbf OVO')
% %% Clinc result
% % preprocessing
% clear all
% tic
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Data_Cancer_Filtered_LeastFreq_Words.mat');
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Train_Label_Cancer.mat');
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Data_Cancer_Filtered_LeastFreq_Words.mat');
% load('../Cancer_Detection_Data_Decreased_Vocab_Size/Test_Label_Cancer');
% vocab = importdata('../Cancer_Detection_Data_Decreased_Vocab_Size/vocabulary_Filtered_LeastFreq_Words.txt');
% toc
% rbf_power = -7:9;
% rbf_sig = 2.^rbf_power;
% boxcon_power = -7:13;
% boxcon = 2.^boxcon_power;
% filename = "Clinc_rbf_OVO";
% function_rbf_OVO(train_data_cancer,test_data_cancer,Train_Label_Cancer,...
%     Test_Label_Cancer,vocab,boxcon,rbf_sig,filename)
% % end
% 
% % %% Report CCR
% %
% % % Confustion matrix based on best CCR
% %
% % %% Report F-score and recall
% %
% % % Confustion matrix based on best F-score and recall
