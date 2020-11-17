clc
clear all
close all

fprintf('\n\n This program accompanies the note entitled "Illustrating the Methodology in EKNR (2015): Some Simple Examples",')
fprintf('\n by Jonathan Eaton, Samuel Kortum, and Brent Neiman. The code solves a simplified 2-country and 2-sector version')
fprintf('\n of the model. It first solves the problem in levels, taking the initial values of the capital stocks in each country')
fprintf('\n as given. Next, it solves the problem in change form, or in "hats", taking the initial values of GDPs as given, as')
fprintf('\n in the method used in the full 21-country and 4-sector model in EKNR (2015). Finally, it verifies that')
fprintf('\n the solution in change form exactly matches that implied by the solution to the standard levels problem, though')
fprintf('\n without information on some of the levels of variables (such as initial capital). For more details and the complete')
fprintf('\n paper, see authors'' webpages. \n\n The code may take several minutes to run and outputs a PDF file called "EKNR_Simple_2Cty.pdf".\n')

%% PROGRAM PARAMETERS
T_data  =   20; % Period for which we observe shocks
T_tail  =   250; % Period where we assume constant shock values
T       =   T_data + T_tail;
fsolve_options_SS = optimoptions( 'fsolve' , 'Display' , 'off' , 'TolFun' , 1e-6 , 'MaxFunEvals' , 1e10 , 'MaxIter' , 100 ) ;
fsolve_options_lev = optimoptions( 'fsolve' , 'Display' , 'off' , 'TolFun' , 1e-5 , 'MaxFunEvals' , 1e10 , 'MaxIter' , 100 ) ;
fsolve_options_hat = optimoptions( 'fsolve' , 'Display' , 'off' , 'TolFun' , 1e-5 , 'MaxFunEvals' , 1e10 , 'MaxIter' , 100 ) ;


%% INITIAL CONDITIONS AND FIXED PARAMETERS
omega = [1/2 1/2]';
K_init_rel_SS = rand(2,1); % Initial Condition on K's (Fraction of SS)
betaL   =   2/3;
betaK   =   1-betaL;
delta   =   0.1;
rho     =   0.95;
theta   =   2;
alpha   =   0.55;


%% HYPOTHETICAL DATA ON SHOCKS (ASSUMED CONSTANT DURING T_TAIL)
load('EKNR_Simple_2Cty_Shocks.mat','chi_data','AD_data','phi_data','L_data','dni_data') % Initial phi values consistent with omega and hypothetical consumption data
chi_tail    =   chi_data(:,T_data)*ones(1,T_tail);
AD_tail     =   AD_data(:,T_data)*ones(1,T_tail);
phi_tail      =   phi_data(:,T_data)*ones(1,T_tail);
L_tail      =   L_data(:,T_data)*ones(1,T_tail);
dni_tail      =   repmat(dni_data(:,:,T_data),1,1,T_tail);
chi =   [chi_data chi_tail];
AD  =   [AD_data AD_tail];
phi =   [phi_data phi_tail];
phi = phi + repmat(ones(1,T)-sum(repmat(omega,1,T).*phi),2,1); % Guarantees admissibility of phi shocks (i.e. sum(omega.*phi(:,t)) always equals one)

L   =   [L_data L_tail];
dni(:,:,1:T_data) = dni_data;
dni(:,:,T_data+1:T) = dni_tail;


%% KEY ANALYTICAL EXPRESSIONS
B  =   betaL^-betaL * betaK^-betaK;
YS_SS = omega.*phi(:,T); % Path of Consumption


%% SOLVE NUMERICALLY FOR STEADY STATE VALUES
fprintf('\n Step 1: Trying to Solve for the Steady State ... ')
[arg_SS,residuals_SS,flag_SS] =  fsolve(@(arg) Fun_2Cty_SS( arg, chi(:,T), AD(:,T), dni(:,:,T), L(:,T), betaL, betaK, delta, rho, theta, alpha, omega.*phi(:,T) ) , [200*YS_SS 20*YS_SS], fsolve_options_SS) ;
if flag_SS<=0 && sum(sum(residuals_SS))>1e-6
    fprintf('\n\n Could Not Solve for the Steady State. Please Try Again.')
%     break
end
fprintf('Successfully Solved for the Steady State\n')
K_SS = arg_SS(:,1);
Y_SS = arg_SS(:,2);
[residuals_SS,XD_SS,YD_SS,pD_SS,piD_SS] = Fun_2Cty_SS( [K_SS Y_SS], chi(:,T), AD(:,T), dni(:,:,T), L(:,T), betaL, betaK, delta, rho, theta, alpha, omega.*phi(:,T) );


%% SOLVE FOR SOLUTION TO PROBLEM IN LEVELS
K_init = K_init_rel_SS.*K_SS;
YS = repmat(omega,1,T).*phi;
fprintf('\n Step 2: Trying to Solve the Problem in Levels ... ')
[Y_levsolution,residuals_lev,flag_lev] =  fsolve(@(Y) Fun_2Cty_Levels( Y, chi, AD, dni, L, betaL, betaK, delta, rho, theta, alpha, YS, K_init, K_SS, T ) , repmat(Y_SS,1,T) , fsolve_options_lev) ;
if flag_lev<=0 && sum(sum(residuals_lev))>1e-5
    fprintf('\n\n Could Not Solve the Problem in Levels. Please Try Again.')
%     break
end
fprintf('Successfully Solved the Problem in Levels \n')
[residual,K_levsolution,piD_levsolution,negative_penalty_levsolution] = Fun_2Cty_Levels( Y_levsolution, chi, AD, dni, L, betaL, betaK, delta, rho, theta, alpha, YS, K_init, K_SS, T );


%% HYPOTHETICAL DATA ON SHOCKS IN CHANGE (OR "HAT") FORM
chi_hat     =   [chi(:,2:T)./chi(:,1:T-1) ones(2,1)];
AD_hat      =   [AD(:,2:T)./AD(:,1:T-1) ones(2,1)];
phi_hat     =   [phi(:,2:T)./phi(:,1:T-1) ones(2,1)];
L_hat       =   [L(:,2:T)./L(:,1:T-1) ones(2,1)];
dni_hat     =   dni(:,:,2:T)./dni(:,:,1:T-1);
dni_hat(:,:,T)  =   ones(2,2,1);


%% SOLVE FOR SOLUTION TO PROBLEM IN "HATS"
fprintf('\n Step 3: Trying to Solve the Problem in Changes ("Hats") ... ')
[hats_solution,residuals_hat,flag_hat] =  fsolve(@(arg) Fun_2Cty_Changes(arg, chi_hat, AD_hat, phi_hat, dni_hat, L_hat, betaL, betaK, delta, rho, theta, alpha, piD_levsolution(:,:,1), Y_levsolution(:,1), YS(:,1), T) , ones(2,T) , fsolve_options_hat) ;
if flag_hat<=0 && sum(sum(residuals_hat))>1e-5
    fprintf('\n\n Could Not Solve the Problem in Changes ("Hats"). Please Try Again.')
%     break
end
fprintf('Successfully Solved the Problem in Changes ("Hats") \n')
[residual,K_hat,Y_hat,negative_penalty_hat] = Fun_2Cty_Changes(hats_solution, chi_hat, AD_hat, phi_hat, dni_hat, L_hat, betaL, betaK, delta, rho, theta, alpha, piD_levsolution(:,:,1), Y_levsolution(:,1), YS(:,1), T);


%% PLOT COMPARISON OF LEVELS AND CHANGES SOLUTIONS (SAVE AS EKNR_Simple_2Cty.pdf)
fprintf('\n Step 4: Plotting Results ... ')
set(figure,'Color',[1 1 1]);
subplot(2,2,1);
handA=plot(1:80,K_levsolution(1,1:80),1:80,K_levsolution(1,1)*cumprod([1 K_hat(1,1:79)],2));
set(handA(1),'LineWidth',3,'LineStyle','--' ,'Color',[1 0 0]);
set(handA(2),'LineWidth',3,'LineStyle',':','Color',[0 0 1]);
legend('Levels Solution','Changes Solution (Normalized at t=1)');
handA=xlabel('Periods');
handA=ylabel('Country 1, Capital Paths');
set(handA, 'FontSize', 20);
subplot(2,2,2);
handB=plot(1:80,K_levsolution(2,1:80),1:80,K_levsolution(2,1)*cumprod([1 K_hat(2,1:79)],2));
set(handB(1),'LineWidth',3,'LineStyle','--' ,'Color',[1 0 0]);
set(handB(2),'LineWidth',3,'LineStyle',':','Color',[0 0 1]);
legend('Levels Solution','Changes Solution (Normalized at t=1)');
handB=xlabel('Periods');
handB=ylabel('Country 2, Capital Paths');
set(handB, 'FontSize', 20);
subplot(2,2,3);
handC=plot(1:80,Y_levsolution(1,1:80),1:80,Y_levsolution(1,1)*cumprod([1 Y_hat(1,1:79)],2));
set(handC(1),'LineWidth',3,'LineStyle','--' ,'Color',[1 0 0]);
set(handC(2),'LineWidth',3,'LineStyle',':','Color',[0 0 1]);
legend('Levels Solution','Changes Solution (Normalized at t=1)');
handC=xlabel('Periods');
handC=ylabel('Country 1, GDP Paths');
set(handC, 'FontSize', 20);
subplot(2,2,4);
handD=plot(1:80,Y_levsolution(2,1:80),1:80,Y_levsolution(2,1)*cumprod([1 Y_hat(2,1:79)],2));
set(handD(1),'LineWidth',3,'LineStyle','--' ,'Color',[1 0 0]);
set(handD(2),'LineWidth',3,'LineStyle',':','Color',[0 0 1]);
legend('Levels Solution','Changes Solution (Normalized at t=1)');
handD=xlabel('Periods');
handD=ylabel('Country 2, GDP Paths');
set(handD, 'FontSize', 20);
orient landscape
print -painters -dpdf -r600 EKNR_Simple_2Cty.pdf
fprintf('Figure Generated and File EKNR_Simple_2Cty.pdf Saved\n')