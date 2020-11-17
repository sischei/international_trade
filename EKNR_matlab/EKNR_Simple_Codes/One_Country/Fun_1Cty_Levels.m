function [residual,K] = Fun_1Cty_Levels( Y, chi, AD, L, betaL, betaK, delta, rho, B, K_SS, K_init, T )
    K(1) = K_init;
    for tt=2:T
        K(tt) = chi(tt-1)*AD(tt-1)*B*K(tt-1)*(K(tt-1)/L(tt-1))^(betaK-1)*(Y(tt-1)-1)/Y(tt-1)+(1-delta)*K(tt-1);
        euler_err(tt-1) =  abs( Y(tt) /  ( Y(tt-1)/(chi(tt-1)*AD(tt-1)*B* K(tt-1))*(K(tt-1)/L(tt-1))^betaL / rho * K(tt) / ((1-delta)/(chi(tt)*AD(tt)*B) * (K(tt)/L(tt))^betaL + betaK) ) - 1 ); 
    end 
    ss_err = abs(K(T)/K_SS-1);
    residual(1:T-1) = euler_err(1:T-1);
    residual(T) = ss_err;
end
