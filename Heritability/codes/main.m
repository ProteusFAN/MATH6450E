%%  configuration

%% ==== path ====
clear;
clc;
path = './';
chdir(path)
addpath(genpath(pwd))

%% ==== preprocessing ====
% G: genotype matrix
% phenotype: phenotype matrix
G = load('GWASgenotype.csv');
phenotype = load('GWASphenotype.csv');

% n: the number of samples
% p: the number of genetic markers
% num_phenotype: the number of phenotype
[n, p] = size(G);
num_phenotype = size(phenotype,2);

% X in LMM
% y: one phenotype
[X, ~, ~] = svd(G);
X = [X(:,1:10), ones(n,1)];
y = phenotype(:,1);

% p_allele: the frequency of the reference allele
% W: standardized genotype matrix
p_allele = sum(G,1)/(2*n);
W = bsxfun(@minus, G, p_allele);
W = bsxfun(@times, W, 1./sqrt(2*p_allele.*(1-p_allele)*p));
K = W*W';
[U,S,~] = svd(K);
S = diag(S);

global paras
paras.UX = U'*X;
paras.Uy = U'*y;
paras.S = S;
paras.n = n;

%% ==== maximum likelihood estimation

delta = fminbnd(@neg_loglikelihoood,exp(1)^(-10), exp(1)^10);

