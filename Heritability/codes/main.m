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

%% ==== MLE & standard error based on fisher information ====
global paras
paras.UX = U'*X;
paras.S = S;
paras.n = n;

% record estimators and standard error
delta = cell(4,1);
beta = cell(4,1);
sigma_u = cell(4,1);
sigma_e = cell(4,1);
heritability = cell(4,1);

fisherInfo = cell(4,1);
beta_se = cell(4,1);
sigma_u_se = cell(4,1);
sigma_e_se = cell(4,1);
heritability_se = cell(4,1);

for phenotype_ind = 1:4
    paras.Uy = U'*phenotype(:,phenotype_ind);
    records = zeros(100,2);
    
    % ==== maximum likelihood estimation ====
    for j = 1:100
        [delta_tmp, val] = fminbnd(@neg_loglikelihoood, exp(-10+0.2*(j-1)), exp(-10+0.2*j));
        records(j,1) = val;
        records(j,2) = delta_tmp; 
        fprintf('phenotype: %d, interval: %d.\n', phenotype_ind, j);
    end
    
    [~, index] = min(records(:,1),[],1);
    delta{phenotype_ind} = records(index(1),2);
    [beta{phenotype_ind}, sigma_u{phenotype_ind}, sigma_e{phenotype_ind}] = parasEst(delta{phenotype_ind});
    
    paras.beta = beta{phenotype_ind};
    paras.sigma_u = sigma_u{phenotype_ind};
    paras.sigma_e = sigma_e{phenotype_ind};
    
    % ==== standard error of MLE based on observed fisher information ====
    fisherInfo = fisherInfoBetaSigma();
    se = sqrt(diag(pinv(fisherInfo)));
    beta_se{phenotype_ind} = se(1:11);
    sigma_u_se{phenotype_ind} = se(12);
    sigma_e_se{phenotype_ind} = se(13);
    
    % ==== compute heritability ====
    heritability{phenotype_ind} = 1/(1+delta{phenotype_ind});
    
    % ==== standard error of heritability based on delta method ====
    gradSigma_ue = [-paras.sigma_e/(paras.sigma_u+paras.sigma_e)^2,paras.sigma_u/(paras.sigma_u+paras.sigma_e)^2]';
    var = gradSigma_ue'*pinv(fisherInfoSigma())*gradSigma_ue;
    heritability_se{phenotype_ind} = sqrt(var);
end
