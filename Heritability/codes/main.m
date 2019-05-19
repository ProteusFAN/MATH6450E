%% Estimatation of Heritability base on Liner Mixed Model 

%% ==== path setting ====
clear; clc;
path = './';
chdir(path)
addpath(genpath(pwd))

%% ==== preprocessing with orignial data ====
% % G: genotype matrix
% % phenotype: phenotype matrix
% G = load('GWASgenotype.csv');
% phenotype = load('GWASphenotype.csv');
% 
% % n: the number of samples
% % p: the number of genetic markers
% % num_phenotype: the number of phenotype
% [n, p] = size(G);
% num_phenotype = size(phenotype,2);
% 
% % X in LMM
% [X, ~, ~] = svd(G);
% X = [X(:,1:10), ones(n,1)];
% y = phenotype(:,1);
% 
% % p_allele: the frequency of the reference allele
% % W: standardized genotype matrix
% p_allele = sum(G,1)/(2*n);
% W = bsxfun(@minus, G, p_allele);
% W = bsxfun(@times, W, 1./sqrt(2*p_allele.*(1-p_allele)*p));
% [U,S,~] = svd(W);
% S = diag(S).^2;

%% ==== load metaData after preprocessing ====
% metaData contains n, num_phenotype, p, phenotype, X, S, U
load('metaData.mat');

%% ==== maximum likelihood estimation ====

global paras
paras.UX = U'*X;
paras.S = S;
paras.n = n;

records = zeros(100,2,4);

for i = 1:4
    paras.Uy = U'*phenotype(:,i);
    for j = 1:100
        [delta, val] = fminbnd(@neg_loglikelihoood, exp(-10+0.2*(j-1)), exp(-10+0.2*j));
        records(j,1,i) = val;
        records(j,2,i) = delta; 
        fprintf('phenotype: %d, interval: %d.\n',i,j);
    end
end

[~, index] = min(records(:,1,:),[],1);

delta = [records(index(1),2,1), records(index(2),2,2), records(index(3),2,3), records(index(4),2,4)];

