clc
clear all
close all

fprintf('\n\n This program accompanies the note entitled "Illustrating the Methodology in EKNR (2015): Some Simple Examples",')
fprintf('\n by Jonathan Eaton, Samuel Kortum, and Brent Neiman. The code solves a simplified closed-economy version of the')
fprintf('\n neoclassical growth model. It first solves the problem in levels, taking the initial value of the capital stock as')
fprintf('\n given, as is standard. Next, it solves the problem in change form, or in "hats", taking the initial value of GDP')
fprintf('\n as given, as in the method used in the full model in EKNR (2015), which has 21 countries and 4 sectors. Finally, it')
fprintf('\n verifies that the solution in change form exactly matches that implied by the solution to the standard levels problem,')
fprintf('\n though without information on some of the levels of variables (such as initial capital). For more details and the')
fprintf('\n complete paper, see authors'' webpages. \n\n The code may take several minutes to run and outputs a PDF file called "EKNR_Simple_1Cty.pdf".\n')


%% PROGRAM PARAMETERS
T_data  =   20; % Period for which we observe shocks
T_tail  =   500; % Period where we assume constant shock values
T       =   T_data + T_tail;
fsolve_options = optimoptions( 'fsolve' , 'Display' , 'off' , 'TolFun' , 1e-6 , 'MaxFunEvals' , 1e10 , 'MaxIter' , 100 ) ;


%% INITIAL CONDITIONS AND FIXED PARAMETERS
K_init_rel_SS = 1/2; % Initial Condition on K's (Fraction of SS)
betaL   =   2/3;
betaK   =   1-betaL;
delta   =   0.1;
rho     =   0.95;
B       =   betaL^-betaL * betaK^-betaK;


%% HYPOTHETICAL DATA ON SHOCKS (ASSUMED CONSTANT DURING T_TAIL)
load('EKNR_Simple_1Cty_Shocks.mat', 'chi_data','AD_data','L_data')
chi_tail    =   chi_data(T_data)*ones(T_tail,1);
AD_tail     =   AD_data(T_data)*ones(T_tail,1);
L_tail      =   L_data(T_data)*ones(T_tail,1);
chi =   [chi_data'; chi_tail];
AD  =   [AD_data'; AD_tail];
L   =   [L_data'; L_tail];


%% STEADY STATE VALUES (FROM ANALYTICAL SOLUTION)
K_SS    =   L(T)*betaK/betaL*(rho*chi(T)*AD(T)/(1-rho*(1-delta)))^(1/betaL);
Y_SS    =   (1-rho*(1-delta))/(1-rho*(1-delta)-delta*rho*betaK);


%% SOLVE FOR SOLUTION TO PROBLEM IN LEVELS
K_init = K_SS * K_init_rel_SS;
Y_guess = ones(T,1)*Y_SS;
fprintf('\n Step 1: Trying to Solve the Problem in Levels ... ')
[Y_levsolution,residuals_levsolution,flag_levsolution] =  fsolve(@(Y) Fun_1Cty_Levels(Y, chi, AD, L, betaL, betaK, delta, rho, B, K_SS, K_init, T) , Y_guess , fsolve_options) ;
if flag_levsolution<=0 && sum(sum(abs(residuals_levsolution)))>1e-6
    fprintf('\n\n Could Not Solve the Problem in Levels. Please Try Again.')
    break
end
fprintf('Successfully Solved the Problem in Levels \n')
[residuals_levsolution,K_levsolution] = Fun_1Cty_Levels(Y_levsolution, chi, AD, L, betaL, betaK, delta, rho, B, K_SS, K_init, T) ;


%% HYPOTHETICAL DATA ON SHOCKS IN CHANGE (OR "HAT") FORM
chi_hat     =   [chi(2:T)./chi(1:T-1); 1];
AD_hat     =   [AD(2:T)./AD(1:T-1); 1];
L_hat     =   [L(2:T)./L(1:T-1); 1];


%% SOLVE FOR SOLUTION TO PROBLEM IN CHANGES (OR "HATS")
hatsolution_guess = ones(T,1); % Guess vector of ones;
fprintf('\n Step 2: Trying to Solve the Problem in Changes (or "Hats") ... ')
[hatsolution,residuals_hatsolution,flag_hatsolution] =  fsolve(@(arg) Fun_1Cty_Changes(arg, chi_hat, AD_hat, L_hat, betaL, betaK, delta, rho, Y_levsolution(1), T) , hatsolution_guess , fsolve_options) ;
if flag_hatsolution<=0 && sum(sum(abs(residuals_hatsolution)))>1e-6
    fprintf('\n\n Could Not Solve the Problem in Changes. Please Try Again.')
    break
end
fprintf('Successfully Solved the Problem in Changes (or "Hats") \n')
[residuals_hatsolution,K_hat_hatsolution,Y_hat_hatsolution] =  Fun_1Cty_Changes(hatsolution, chi_hat, AD_hat, L_hat, betaL, betaK, delta, rho, Y_levsolution(1), T);


%% PLOT LEVELS AND CHANGE SOLUTIONS (SAVE AS EKNR_Simple_1Cty.pdf)
fprintf('\n Step 3: Plotting Results ... ')
set(figure,'Color',[1 1 1]);
subplot(1,2,1);
handA=plot(1:80,K_levsolution(1:80),1:80,K_levsolution(1)*cumprod([1 K_hat_hatsolution(1:79)]));
set(handA(1),'LineWidth',3,'LineStyle','--' ,'Color',[1 0 0]);
set(handA(2),'LineWidth',3,'LineStyle',':','Color',[0 0 1]);
legend('Levels Solution','Changes Solution (Normalized at t=1)');
handA=xlabel('Periods');
handA=ylabel('Path of Capital');
set(handA, 'FontSize', 20);
subplot(1,2,2);
handB=plot(1:80,Y_levsolution(1:80),1:80,Y_levsolution(1)*cumprod([1 Y_hat_hatsolution(1:79)]));
set(handB(1),'LineWidth',3,'LineStyle','--' ,'Color',[1 0 0]);
set(handB(2),'LineWidth',3,'LineStyle',':','Color',[0 0 1]);
legend('Levels Solution','Changes Solution (Normalized at t=1)');
handB=xlabel('Periods');
handB=ylabel('Path of GDP');
set(handB, 'FontSize', 20);
orient landscape
print -painters -dpdf -r600 EKNR_Simple_1Cty.pdf
fprintf('Figure Generated and File EKNR_Simple_1Cty.pdf Saved\n')


