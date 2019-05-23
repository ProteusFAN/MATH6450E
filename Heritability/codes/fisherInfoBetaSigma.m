function fisherInfo = fisherInfoBetaSigma()
%% compute fisher information matrix of [beta, sigma_u, sigma_e]
global paras

info_beta = fisherInfoBeta();
info_sigma = fisherInfoSigma();

info_beta_sigma = zeros(11,2);
tmp_1 = 0;
tmp_2 = 0;
for i = 1:11
    for j = 1:paras.n
        tmp_beta = (paras.UX(j,i)*(paras.Uy(j) - paras.UX(j,:)*paras.beta)/(paras.S(j)*paras.sigma_u + paras.sigma_e));
        tmp_u = -1/2*(paras.S(j)/(paras.S(j)*paras.sigma_u+paras.sigma_e)-paras.S(j)*((paras.Uy(j)-paras.UX(j,:)*paras.beta)/(paras.S(j)*paras.sigma_u+paras.sigma_e))^2);
        tmp_e = -1/2*(1/(paras.S(i)*paras.sigma_u+paras.sigma_e)-((paras.Uy(i)-paras.UX(i,:)*paras.beta)/(paras.S(i)*paras.sigma_u+paras.sigma_e))^2);
        tmp_1 = tmp_1 + tmp_beta * tmp_u;
        tmp_2 = tmp_2 + tmp_beta * tmp_e;
    end
    info_beta_sigma(i,:) = [tmp_1, tmp_2];
end

fisherInfo = [info_beta, info_beta_sigma; info_beta_sigma', info_sigma];

end