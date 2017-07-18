%% configure /online-e-svm

% setup path
addpath('./src')

% setup mc reader
mc_path = './matrix_chain/';
addpath(mc_path)

% setup incomplete Cholesky
cd ./cholesky/
mex incchol.c
cd ..
addpath('./cholesky/');