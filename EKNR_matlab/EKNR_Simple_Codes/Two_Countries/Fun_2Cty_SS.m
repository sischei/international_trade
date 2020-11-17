function [residual,XD_SS,YD_SS,pD_SS,piD_SS] = Fun_2Cty_SS( arg, chi_SS, AD_SS, dni_SS, L_SS, betaL, betaK, delta, rho, theta, alpha, YS_SS )
    K_SS = arg(:,1);
    Y_SS = arg(:,2);
    YD_SS = Y_SS - YS_SS;
    w_SS = betaL.*Y_SS./L_SS;
    r_SS = betaK.*Y_SS./K_SS;
    b_SS = w_SS.^betaL.*r_SS.^betaK;
    pD_SS(1)  = ((b_SS(1).*dni_SS(1,1)./AD_SS(1)).^-theta + (b_SS(2).*dni_SS(1,2)./AD_SS(2)).^-theta).^(-1/theta);
    pD_SS(2)  = ((b_SS(1).*dni_SS(2,1)./AD_SS(1)).^-theta + (b_SS(2).*dni_SS(2,2)./AD_SS(2)).^-theta).^(-1/theta);
    piD_SS(1:2,1)  = (b_SS(1).*dni_SS(1:2,1)./pD_SS(1:2)'./AD_SS(1)).^-theta;
    piD_SS(1:2,2)  = (b_SS(2).*dni_SS(1:2,2)./pD_SS(1:2)'./AD_SS(2)).^-theta;
    XD_SS = inv(piD_SS')*YD_SS; 
    ss_err1 = abs(XD_SS./pD_SS'./K_SS./(delta./chi_SS).^(1/alpha)-1) + 1e5*(XD_SS./pD_SS'./K_SS - (delta./chi_SS).^(1/alpha)./2).^2 .* (XD_SS./pD_SS'./K_SS <= (delta./chi_SS).^(1/alpha)./2);
    ss_err2 = abs(pD_SS'./chi_SS.*(1-rho*(1-delta))./rho./(XD_SS./pD_SS'./K_SS).^(alpha)./(pD_SS'.*(1-alpha)+betaK.*Y_SS./K_SS.*alpha./(XD_SS./pD_SS'./K_SS)) - 1 );
    residual = [ss_err1 ss_err2];
end
