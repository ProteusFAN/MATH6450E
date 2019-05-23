function info_beta = fisherInfoBeta()
%% compute fisher information matrix of beta
global paras

info_beta = zeros(11,11);
for i = 1:11
    for j = 1:11
        tmp = 0;
        % for iter = 1:paras.n
        %     tmp = tmp + paras.UX(iter,i)*paras.UX(iter,j)*((paras.Uy(iter) - paras.UX(iter,:)*paras.beta)/(paras.S(iter)*paras.sigma_u + paras.sigma_e))^2;
        % end
        for iter = 1:paras.n
            tmp = tmp + paras.UX(iter,i)*paras.UX(iter,j)/(paras.S(iter)*paras.sigma_u + paras.sigma_e);
        end
        info_beta(i,j) = tmp;
    end
end

end