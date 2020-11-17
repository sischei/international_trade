function [residual,K_hat,Y_hat] = Fun_1Cty_Changes( solution_hats, chi_hat, AD_hat, L_hat, betaL, betaK, delta, rho, Y_init, T )
    K_hat(1) = solution_hats(1);
    Y_hat(1:T-1) = solution_hats(2:T);
    Y(1) = Y_init;
    for tt=2:T
        Y(tt) = Y(tt-1)*Y_hat(tt-1);
    end
    for tt=2:T
        %Note: Y(tt) corresponds to the period before Y_hat(tt).
        euler_err(tt-1) =  abs( Y_hat(tt-1) / ( K_hat(tt-1) / rho / ((1-delta)/(chi_hat(tt-1)*AD_hat(tt-1)) * (K_hat(tt-1)/L_hat(tt-1))^betaL + betaK*(K_hat(tt-1)-(1-delta))*(Y(tt-1)/(Y(tt-1)-1)) ) ) - 1 ); 
        K_hat(tt) = chi_hat(tt-1)*AD_hat(tt-1)*(K_hat(tt-1)/L_hat(tt-1))^(betaK-1)*(Y(tt-1)*Y_hat(tt-1)-1)/((Y(tt-1)-1)*Y_hat(tt-1))*(K_hat(tt-1)-(1-delta))+(1-delta);
    end 
    ss_err = abs(K_hat(T)/1-1);
    residual(1:T-1) = euler_err(1:T-1);
    residual(T) = ss_err;
end
