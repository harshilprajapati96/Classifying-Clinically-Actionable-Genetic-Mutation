%% %% SVM Classifier for Text Document
% MATLAB R2017b
% Bowen Song U04079758
casenumber = input('Enter a number: ');

switch casenumber
    case 1
        disp('linear News 20 group OVA')
%% News 20 group OVA
% preprocessing
% tic
clear all
Preprocessing_new20;
disp("Preprocessing is done:")
% toc
boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "USnews_linear_OVA";
function_linear_OVA(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,filename)
    case 2
        disp('linear News 20 group OVO')
%% News 20 group OVO
% preprocessing
clear all
Preprocessing_new20;
disp("Preprocessing is done:")

boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "USnews_linear_OVO";
function_linear_OVO(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,filename)
    case 3
        disp('linear Clinic OVA')
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
    case 4
        disp('linear Clinic OVO')
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
filename = "Clinc_linear_OVO";
function_linear_OVO(train_data_cancer,test_data_cancer,Train_Label_Cancer,...
    Test_Label_Cancer,vocab,boxcon,filename)


    case 5
        disp('News 20 group OVA')
%% News 20 group OVA
% preprocessing
% tic
clear all
Preprocessing_new20;
disp("Preprocessing is done:")
% toc
% boxcon_power = -7:13;
% boxcon = 2.^boxcon_power;
% filename = "USnews_linear_OVA";
% function_linear_OVA(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,filename)
    case 6
        disp('News 20 group OVO')
%% News 20 group OVO
% preprocessing
clear all
Preprocessing_new20;
disp("Preprocessing is done:")

boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "USnews_linear_OVA";
function_linear_OVA(X_train_woSTOP,X_test_woSTOP,Y_train,Y_test,vocab,boxcon,filename)
    case 7
        disp('Clinic OVA')
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
    case 8
        disp('Clinic OVO')
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
    case 9
        disp('Clinic OVO')
%% Clinc result
% preprocessing
clear all
Preprocessing_new20;

X_train_woSTOP = X_train_woSTOP(ismember(Y_train_expand,[1,20]),:);
X_test_woSTOP = X_test_woSTOP(ismember(Y_test_expand,[1,20]),:);
Y_train = Y_train(ismember(Y_train,[1,20]));
Y_test = Y_test(ismember(Y_test,[1,20]));

boxcon_power = -7:13;
boxcon = 2.^boxcon_power;
filename = "Clinc_linear_OVO";
function_linear_OVO(train_data_cancer,test_data_cancer,Train_Label_Cancer,...
    Test_Label_Cancer,vocab,boxcon,filename,toy)
end

% %% Report CCR
% 
% % Confustion matrix based on best CCR
% 
% %% Report F-score and recall
% 
% % Confustion matrix based on best F-score and recall
