function info_sigma = fisherInfoSigma()
%% compute fisher information matrix of [sigma_u, sigma_e]
global paras

var_u = 0;
for i = 1:paras.n
    tmp = paras.S(i)/(paras.S(i)*paras.sigma_u+paras.sigma_e)-paras.S(i)*((paras.Uy(i)-paras.UX(i,:)*paras.beta)/(paras.S(i)*paras.sigma_u+paras.sigma_e))^2;
    var_u = var_u + 1/4 * tmp^2;
end

var_e = 0;
for i = 1:paras.n
    tmp = 1/(paras.S(i)*paras.sigma_u+paras.sigma_e)-((paras.Uy(i)-paras.UX(i,:)*paras.beta)/(paras.S(i)*paras.sigma_u+paras.sigma_e))^2;
    var_e = var_e + 1/4 * tmp^2;
end

cov = 0;
for i = 1:paras.n
    tmp_u = paras.S(i)/(paras.S(i)*paras.sigma_u+paras.sigma_e)-paras.S(i)*((paras.Uy(i)-paras.UX(i,:)*paras.beta)/(paras.S(i)*paras.sigma_u+paras.sigma_e))^2;
    tmp_e = 1/(paras.S(i)*paras.sigma_u+paras.sigma_e)-((paras.Uy(i)-paras.UX(i,:)*paras.beta)/(paras.S(i)*paras.sigma_u+paras.sigma_e))^2;
    cov = cov + 1/4 * tmp_u * tmp_e;
end

info_sigma = [var_u, cov; cov, var_e];

end