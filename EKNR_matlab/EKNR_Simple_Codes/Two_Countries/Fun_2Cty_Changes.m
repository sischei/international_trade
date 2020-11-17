function [residual,K_hat,Y_hat,negative_penalty] = Fun_2Cty_Changes( solution_hats, chi_hat, AD_hat, phi_hat, dni_hat, L_hat, betaL, betaK, delta, rho, theta, alpha, piD_init, Y_init, YS_init, T )
	K_hat(1:2,1) = solution_hats(:,1);
    Y_hat(1:2,1:T-1) = solution_hats(:,2:T);
    Y(1:2,1) = Y_init;
    YS(1:2,1) = YS_init;
    YD(1:2,1) = Y(:,1)-YS(:,1);
    piD(:,:,1) = piD_init;
    XD(:,1) = inv(piD(:,:,1)')*YD(:,1);
    for tt=2:T
        Y(:,tt) = Y(:,tt-1).*Y_hat(:,tt-1);
        YS(:,tt) = YS(:,tt-1).*phi_hat(:,tt-1);
        YD(:,tt) = Y(:,tt) - YS(:,tt);
        YS_hat(:,tt-1) = YS(:,tt)./YS(:,tt-1);
        YD_hat(:,tt-1) = YD(:,tt)./YD(:,tt-1);
        w_hat(1:2,tt-1) = Y_hat(:,tt-1)./L_hat(:,tt-1);
        r_hat(1:2,tt-1) = Y_hat(:,tt-1)./K_hat(:,tt-1);
        b_hat(1:2,tt-1) = w_hat(:,tt-1).^betaL.*r_hat(:,tt-1).^betaK;
        pD_hat(1,tt-1)  = (piD(1,1,tt-1).*(b_hat(1,tt-1).*dni_hat(1,1,tt-1)./AD_hat(1,tt-1)).^-theta + piD(1,2,tt-1).*(b_hat(2,tt-1).*dni_hat(1,2,tt-1)./AD_hat(2,tt-1)).^-theta).^(-1/theta);
        pD_hat(2,tt-1)  = (piD(2,1,tt-1).*(b_hat(1,tt-1).*dni_hat(2,1,tt-1)./AD_hat(1,tt-1)).^-theta + piD(2,2,tt-1).*(b_hat(2,tt-1).*dni_hat(2,2,tt-1)./AD_hat(2,tt-1)).^-theta).^(-1/theta);
        piD_hat(1,1,tt-1) = (b_hat(1,tt-1).*dni_hat(1,1,tt-1)./pD_hat(1,tt-1)./AD_hat(1,tt-1)).^-theta;
        piD_hat(2,1,tt-1) = (b_hat(1,tt-1).*dni_hat(2,1,tt-1)./pD_hat(2,tt-1)./AD_hat(1,tt-1)).^-theta;
        piD_hat(1,2,tt-1) = (b_hat(2,tt-1).*dni_hat(1,2,tt-1)./pD_hat(1,tt-1)./AD_hat(2,tt-1)).^-theta;
        piD_hat(2,2,tt-1) = (b_hat(2,tt-1).*dni_hat(2,2,tt-1)./pD_hat(2,tt-1)./AD_hat(2,tt-1)).^-theta;
        piD(:,:,tt) = piD(:,:,tt-1).*piD_hat(:,:,tt-1);
        XD(:,tt) = inv(piD(:,:,tt)')*YD(:,tt);
        XD_hat(:,tt-1) = XD(:,tt)./XD(:,tt-1);
        negative_penalty(:,tt-1) = (XD_hat(:,tt-1) - 0.2).^2 .* (XD_hat(:,tt-1) <= 0.2);
        XD_hat(:,tt-1) = XD_hat(:,tt-1).*(XD_hat(:,tt-1) > 0.2) + 0.2*(XD_hat(:,tt-1) <= 0.2);
        euler_err(:,tt-1) = abs ( 1/rho*K_hat(:,tt-1)./(K_hat(:,tt-1)-(1-delta)) ./ (XD_hat(:,tt-1).*((1-alpha)+1./chi_hat(:,tt-1).*(K_hat(:,tt-1).*pD_hat(:,tt-1)./XD_hat(:,tt-1)).^alpha.*(1-delta)./(K_hat(:,tt-1)-(1-delta)))+alpha.*betaK.*Y(:,tt)./XD(:,tt-1)) - 1 ); 
        K_hat(:,tt) = (1-delta) + chi_hat(:,tt-1).*(XD_hat(:,tt-1)./pD_hat(:,tt-1)./K_hat(:,tt-1)).^alpha.*(K_hat(:,tt-1)-(1-delta));
    end
    ss_err = abs(K_hat(:,T)./1-1);
    residual(:,1:T-1) = euler_err + negative_penalty;
    residual(:,T) = ss_err;
    residual = residual;
end
