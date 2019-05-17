function ell = neg_loglikelihoood(delta)
%% neg_loglikelihood function w.r.t. delta
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

% compute loglikelihoood
ell = 0;
for i = 1:paras.n
    ell = ell + log(paras.S(i) + delta);
end

tmp = 0;
for i = 1:paras.n
    tmp = tmp + (paras.Uy(i) - paras.UX(i,:) * beta)^2/(paras.S(i) + delta);
end
ell = ell + paras.n * log(1/paras.n * tmp);

end