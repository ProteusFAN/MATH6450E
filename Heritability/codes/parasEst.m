function [beta, sigma_u, sigma_e] = parasEst(delta)
%% parameters estimation given delta
global paras

% compute beta
tmp_inv = zeros(11,11);
for i = 1:paras.n
    tmp_inv = tmp_inv + 1/(paras.S(i) + delta) * paras.UX(i,:)' * paras.UX(i,:);
end
tmp = zeros(11,1);
for i = 1:paras.n
    tmp = tmp + 1/(paras.S(i) + delta) * paras.UX(i,:)' * paras.Uy(i);
end
beta = pinv(tmp_inv) * tmp;

% compute delta_u
tmp = 0;
for i = 1:paras.n
    tmp = tmp + (paras.Uy(i) - paras.UX(i,:) * beta)^2/(paras.S(i) + delta);
end
sigma_u = 1/paras.n * tmp;

% compute delta_e
sigma_e = delta * sigma_u;

end