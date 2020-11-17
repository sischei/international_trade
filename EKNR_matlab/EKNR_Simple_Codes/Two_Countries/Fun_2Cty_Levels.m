function [residual,K,piD,negative_penalty] = Fun_2Cty_Levels( Y, chi, AD, dni, L, betaL, betaK, delta, rho, theta, alpha, YS, K_init, K_SS, T )
    YD = Y - YS;
    K(1:2,1) = K_init;
    w(1:2,1) = betaL.*Y(:,1)./L(:,1);
    r(1:2,1) = betaK.*Y(:,1)./K(:,1);
    b(1:2,1) = w(:,1).^betaL.*r(:,1).^betaK;
    pD(1,1)  = ((b(1,1).*dni(1,1,1)./AD(1,1)).^-theta + (b(2,1).*dni(1,2,1)./AD(2,1)).^-theta).^(-1/theta);
    pD(2,1)  = ((b(1,1).*dni(2,1,1)./AD(1,1)).^-theta + (b(2,1).*dni(2,2,1)./AD(2,1)).^-theta).^(-1/theta);
    piD(1:2,1,1)  = (b(1,1).*dni(1:2,1,1)./pD(1:2,1)./AD(1,1)).^-theta;
    piD(1:2,2,1)  = (b(2,1).*dni(1:2,2,1)./pD(1:2,1)./AD(2,1)).^-theta;
    XD(:,1) = inv(piD(:,:,1)')*YD(:,1);
    negative_penalty(:,1) = (XD(:,1)./pD(:,1)./K(:,1) - (delta./chi(:,1)).^(1/alpha)./2).^2 .* (XD(:,1)./pD(:,1)./K(:,1) <= (delta./chi(:,1)).^(1/alpha)./2);
    XD(:,1) = XD(:,1).*(XD(:,1)./pD(:,1)./K(:,1) > (delta./chi(:,1)).^(1/alpha)./2) + (delta./chi(:,1)).^(1/alpha)./2.*K(:,1).*pD(:,1).*(XD(:,1)./pD(:,1)./K(:,1) <= (delta./chi(:,1)).^(1/alpha)./2);
    for tt=2:T
        K(:,tt) = chi(:,tt-1).*(XD(:,tt-1)./pD(:,tt-1)).^alpha.*K(:,tt-1).^(1-alpha)+(1-delta).*K(:,tt-1);
        w(1:2,tt) = betaL.*Y(:,tt)./L(:,tt);
        r(1:2,tt) = betaK.*Y(:,tt)./K(:,tt);
        b(1:2,tt) = w(:,tt).^betaL.*r(:,tt).^betaK;
        pD(1,tt)  = ((b(1,tt).*dni(1,1,tt)./AD(1,tt)).^-theta + (b(2,tt).*dni(1,2,tt)./AD(2,tt)).^-theta).^(-1/theta);
        pD(2,tt)  = ((b(1,tt).*dni(2,1,tt)./AD(1,tt)).^-theta + (b(2,tt).*dni(2,2,tt)./AD(2,tt)).^-theta).^(-1/theta);
        piD(1:2,1,tt)  = (b(1,tt).*dni(1:2,1,tt)./pD(1:2,tt)./AD(1,tt)).^-theta;
        piD(1:2,2,tt)  = (b(2,tt).*dni(1:2,2,tt)./pD(1:2,tt)./AD(2,tt)).^-theta;
        XD(:,tt) = inv(piD(:,:,tt)')*YD(:,tt); 
        negative_penalty(:,tt) = (XD(:,tt)./pD(:,tt)./K(:,tt) - (delta./chi(:,tt)).^(1/alpha)./2).^2 .* (XD(:,tt)./pD(:,tt)./K(:,tt) <= (delta./chi(:,tt)).^(1/alpha)./2);
        XD(:,tt) = XD(:,tt).*(XD(:,tt)./pD(:,tt)./K(:,tt) > (delta./chi(:,tt)).^(1/alpha)./2) + (delta./chi(:,tt)).^(1/alpha)./2.*K(:,tt).*pD(:,tt).*(XD(:,tt)./pD(:,tt)./K(:,tt) <= (delta./chi(:,tt)).^(1/alpha)./2);
        euler_err(:,tt-1) = abs ( pD(:,tt-1)./alpha./chi(:,tt-1).*(XD(:,tt-1)./pD(:,tt-1)./K(:,tt-1)).^(1-alpha) ./ rho ./ (pD(:,tt)./alpha./chi(:,tt).*(XD(:,tt)./pD(:,tt)./K(:,tt)).^(1-alpha).*(chi(:,tt).*(1-alpha).*(XD(:,tt)./pD(:,tt)./K(:,tt)).^alpha + (1-delta)) + r(:,tt)) - 1 );
    end
    ss_err = abs((chi(:,T).*(XD(:,T)./pD(:,T)).^alpha.*K(:,T).^(1-alpha)+(1-delta).*K(:,T))./K_SS-1);
    residual(:,1:T-1) = euler_err;
    residual(:,T) = ss_err;
    residual = residual + negative_penalty;
end
