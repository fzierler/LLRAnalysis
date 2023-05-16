import mpmath as mp
import numpy as np
import re
import pandas as pd
import os.path
import llranalysis.utils as utils
import llranalysis.standard as standard
import llranalysis.doubleGaussian as dg
def ReadRep(file, new_rep, poly):
    txt_dS0 = "LLR Delta S "
    txt_swap = "New Rep Par S0 = "
    txt_S0 = "LLR S0 Central action"
    txt_a = "a_rho"
    txt_end = "Fixed a MC Step"
    txt_V = "Global size is"
    txt_Plaq = "Plaq a fixed"
    
    txt_fxda_E = "Fixed a MC Step"
    txt_fxda_py = "(Polyakov direction 0 =|FUND_POLYAKOV)"
    txt_fxda_S0 = "New LLR Param: "
    
    
    new_V = 1.
    new_Lt = 1.
    V = []
    Lt = []
    rep = []
    new_dS0 = 0.
    dS0 = []
    a = []
    S0 = []
    new_S0 = 0.
    swap_S0 = []
    rm_end = False
    rm_plaq = []
    new_plaq = 0.
    
    new_a = 0.
    new_E = 0.
    fa_a = []
    fa_S0 = []
    fa_E = []
    fa_py = []
    fa_py2 = []
    fa_rep = []
    fa_V = []
    fa_n = 0
    fa_N = []
    fa_Lt = []
    with open(file, "r") as a_file:
        
        for line in a_file:  
            if(not rm_end):
                #Check for starting action
                if(re.findall(txt_S0, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            new_S0 = float(word)
                elif(re.findall(txt_Plaq, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            new_plaq = float(word)
                elif(re.findall(txt_V, line) != []):
                    new_V = 1.
                    is_Lt = True
                    for word in re.split("x",re.split("\s",line)[-2]):
                        
                        if(utils.check_float(word)):
                            new_V *= float(word)
                            if is_Lt: 
                                new_Lt = float(word)
                                is_Lt = False
                        
                #Check for swap     
                elif(re.findall(txt_swap, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            if(new_S0 != float(word)):
                                new_S0 = float(word)
                                swap_S0.append(new_S0)
                            break
                elif(re.findall(txt_dS0, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            new_dS0 = float(word) 
                #Append a and S0        
                elif(re.findall(txt_a, line) != []):
                    for word in re.split("\s", line):
                        if(utils.check_float(word)):
                            a.append(float(word))
                            S0.append(new_S0)
                            dS0.append(new_dS0)
                            rep.append(new_rep)
                            V.append(new_V)
                            rm_plaq.append(new_plaq * 6. * new_V)
                            Lt.append(new_Lt)
                #End of RM
                elif(re.findall(txt_end, line) != []):
                    new_a = a[-1]
                    rm_end = True
                    for word in re.split("=", line):
                        if(utils.check_float(word)):
                            new_E = float(word)
                            if(not poly):
                                print("Shouldnt be here")
                                fa_S0.append(new_S0)
                                fa_a.append(new_a)
                                fa_E.append(new_E)
            else:
                if(re.findall(txt_fxda_E, line) != []):
                    for word in re.split("=", line):
                        if(utils.check_float(word)):
                            new_E = float(word)
                            if(not poly):
                                print("Shouldnt be here")
                                fa_S0.append(new_S0)
                                fa_a.append(new_a)
                                fa_E.append(new_E)
                elif((len(re.findall(txt_fxda_py, line)) > 1) and poly): 
                    py_tmp = 0.
                    for word in re.split("\s",re.split("=", line)[1]):
                        if(utils.check_float(word)):
                            py_tmp += float(word) ** 2
                    fa_py.append(py_tmp ** 0.5)
                    fa_py2.append(py_tmp)
                    fa_S0.append(new_S0)
                    fa_a.append(new_a)
                    fa_E.append(new_E)
                    fa_V.append(new_V)
                    fa_rep.append(new_rep)
                    fa_N.append(fa_n)
                    fa_n += 1
                    fa_Lt.append(new_Lt)
                elif(re.findall(txt_fxda_S0, line) != []):
                    bool_E = True
                    for word in re.split(r"[,\s]\s*", line):
                        if(utils.check_float(word) and bool_E):
                            new_S0 = float(word)
                            bool_E = False
                        elif(utils.check_float(word) and (not bool_E)):
                            new_a = float(word)
                            break;
    
    fa_S0 = np.array(fa_S0)
    fa_E = np.array(fa_E)
    S0 = np.array(S0) 
    dS0 = np.array(dS0) /  (2)
    a = - 1. * np.array(a) 
    fa_a = - 1. * np.array(fa_a)           
    if poly:
        fixed_a = pd.DataFrame(data = {'a':fa_a,'Ek':fa_S0,'S':fa_E, 'V':fa_V, 'Rep':fa_rep,'Poly':fa_py, 'PolySqr':fa_py2, 'n':fa_N})
    else:
        fixed_a = pd.DataFrame(data = {'a':fa_a,'Ek':fa_S0,'S':fa_E, 'V':fa_V, 'Rep':fa_rep, 'n':fa_N, 'Lt':fa_Lt})
    RM = pd.DataFrame(data = {'n':range(len(a)),'a':a,'Ek':S0,'dE':dS0, 'V':V, 'S': rm_plaq,'Rep':rep, 'Lt':Lt})
    final = pd.DataFrame(data = {'a':a[-1],'Ek':S0[-1],'dE':dS0[-1], 'V':V[-1], 'Lt':new_Lt}, index = [new_rep])
    return RM, fixed_a, final

def ReadFull(files, poly=True):
    fxa_df = pd.DataFrame()
    RM_df = pd.DataFrame()
    final_df = pd.DataFrame()
    for i, file in zip(range(len(files)),files):
        print(file)
        RM_tmp, fxa_tmp, final_tmp = ReadRep(file, i, poly=True)
        if final_df.empty:
            RM_df= RM_tmp.copy()
            fxa_df = fxa_tmp.copy()
            final_df = final_tmp.copy()
        else:       
            RM_df = RM_df.append(RM_tmp, ignore_index=True)
            fxa_df= fxa_df.append(fxa_tmp, ignore_index=True)
            final_df = final_df.append(final_tmp, ignore_index=True)
    RM_df = RM_df.sort_values(by=['Ek','n'], ignore_index = True)
    fxa_df = fxa_df.sort_values(by=['Ek','Rep'], ignore_index = True)
    final_df = final_df.sort_values(by=['Ek'], ignore_index = True)
    print(final_df)
    dE_new = (final_df['Ek'].values[1]-final_df['Ek'].values[0])
    final_df = final_df.assign(dE = np.ones(np.size(final_df['dE'].values)) * dE_new)
    final_df = final_df.assign(Ek = final_df['Ek'].values)
    RM_df = RM_df.assign(Ek = RM_df['Ek'].values)
    fxa_df = fxa_df.assign(Ek = fxa_df['Ek'].values)
    return fxa_df, RM_df, final_df

def SaveCSVFull(files,folder,poly=True):
    FA, RM, FNL = ReadFull(files, poly)
    RM.to_csv(folder + 'RM.csv', index = False)
    FA.to_csv(folder + 'fa.csv', index = False)
    FNL.to_csv(folder + 'final.csv', index = False)
    
def ReadCSVFull(folder):
    RM = pd.read_csv(folder + 'RM.csv')
    FA = pd.read_csv(folder + 'fa.csv')
    FNL = pd.read_csv(folder + 'final.csv')
    return RM, FA, FNL

def CSV(files,folder, poly=True):
    exists = os.path.isfile(folder + 'RM.csv') and os.path.isfile(folder + 'fa.csv') and os.path.isfile(folder + 'final.csv')
    if exists:
        print('Reading csv files')
        RM, FA, FNL = ReadCSVFull(folder)
    else:
        print('No CSV files creating them now')
        SaveCSVFull(files,folder,poly)
        RM, FA, FNL = ReadCSVFull(folder)
    return RM, FA, FNL

def ReadObservables(betas, final_df, fa_df, folder, file = 'obs.csv', calc_poly = True):
    print('Obs_DF')
    file_loc = folder + file
    exists = os.path.isfile(file_loc) 
    if exists:
        obs_DF = pd.read_csv(file_loc)
    else:
        obs_DF = calc_observables_full(betas,final_df,fa_df, calc_poly=calc_poly)
        obs_DF.to_csv(file_loc, index = False)
    obs_DF = obs_DF.sort_values(by=['b'], ignore_index=True)
    obs_DF.to_csv(file_loc, index = False)
    obs_DF = obs_DF[np.in1d(obs_DF['b'].values, betas)].reset_index()
    return obs_DF

def calc_observables_full(betas,final_df,fa_df, calc_poly = True):
    Eks = final_df['Ek'].values
    dE = (Eks[1] - Eks[0]) 
    V = final_df['V'].values[0]
    Lt = final_df['Lt'].values[0]
    aks = final_df['a'].values
    E = []
    E2 = []
    E4 = []
    poly = []
    poly_sq = []
    poly_4 = []
    for beta in betas: 
        print("Beta:",beta)
        E.append((calc_E(Eks, aks, beta)) / (6*V))
        E2.append((calc_E2(Eks, aks, beta)) / ((6*V) ** 2.))
        E4.append((calc_E4(Eks, aks, beta))/ ((6*V) ** 4.))
        if len(fa_df['S']) > 0 and calc_poly:
            lnz_obs = calc_fxa_lnZ(fa_df, final_df, beta)
            poly.append(calc_fxa_obs(fa_df, final_df, beta, 'Poly', 1., lnz_obs))
            poly_sq.append(calc_fxa_obs(fa_df, final_df, beta, 'Poly', 2.,lnz_obs))
            poly_4.append(calc_fxa_obs(fa_df, final_df, beta, 'Poly', 4.,lnz_obs))
        else:
            poly.append(0.)
            poly_sq.append(0.)
            poly_4.append(0.)
    SH =( np.array(E2) - (np.array(E)**2.)) * 6. * V 
    Xlp = (np.array(poly_sq) - (np.array(poly) ** 2.)) * (V/Lt)
    Blp = 0.
    binder = 1 - (np.array(E4) / (3 * (np.array(E2) ** 2.)))
    if(len(poly_sq) > 0): 
        if (min(poly_sq) != 0.): Blp = 1. - (np.array(poly_4) / (3*(np.array(poly_sq) ** 2.))) 
    return pd.DataFrame(data = {'b':betas,'u':E,'Cu':SH,'Bv':binder, 'lp':poly, 'Xlp':Xlp,'Blp':Blp, 'V':V * np.ones(len(poly)), 'Lt':Lt * np.ones(len(poly))})

def calc_lnZ(Eks, aks, beta, rmpf = False):
    pi_exp = 0.
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    Z = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta) 
        if not (A == mp.mpf(0.)):
            full_exp = mp.exp(pi_exp + (Ek*beta) + a*(dE/2.))
            T = A * dE / 2.
            shn = mp.sinh(T)
            Z += full_exp * shn / A
        else:
            print('A=0, beta =',beta,', a=', a)
            full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            Z += (dE / 2.) * full_exp
        pi_exp += a*dE
    ln_Z = mp.ln(2*Z)
    if not rmpf: ln_Z = float(ln_Z)
    return ln_Z

def calc_lnrho(final_df, E):
    pi_exp = 0.
    ln_rho = 0.
    dE = final_df['dE'][0]
    final_df = final_df.sort_values(by=['Ek'], ignore_index=True)
    for En, an in zip(final_df['Ek'].values - (final_df['dE'].values/2), final_df['a'].values):
        if(E >= En and E < (En + dE)):
            ln_rho = (an * (E - En)) + pi_exp
            break
        else:
            pi_exp += an*dE
    if ln_rho == 0:print("Energy not in range:",E)
    return ln_rho

def calc_E(Eks, aks, beta):
    pi_exp = 0.
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    E = mp.mpf(0)
    Z = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        if not (A == mp.mpf(0.)):
            full_exp = mp.exp(pi_exp + (Ek*beta) + a*(dE/2.))
            T = A * dE / 2.
            shn = mp.sinh(T)
            Z += full_exp * shn / A
            E += full_exp *( (Ek - (1./A))*shn + (dE/2.)*mp.cosh(T)) / A
        else:
            print('A=0, beta =',beta,', a=', a)
            full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            E += full_exp * (dE / 2.) * Ek
            Z += (dE / 2.) * full_exp
        pi_exp += a*dE
    return E/Z

def calc_E2(Eks, aks, beta):
    pi_exp = - calc_lnZ(Eks, aks, beta, rmpf = True)
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    E2 = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        if not (A == mp.mpf(0.)):
            full_exp = 2. * mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.))/ A
            T = A * dE / 2.
            E2 += full_exp *(mp.sinh(T)*((2./(A**2.)) - (2.*Ek/A) + ((Ek**2.) + ((dE/2.)**2.))) + mp.cosh(T)*((Ek*dE) - (dE/A))) 
        else:
            print('A=0, beta =',beta,', a=', a)
            full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            E2 += 2 * full_exp * (((Ek**2.) * dE/2.) + ((1./3.) * ((dE/2.) ** 3.)))
        pi_exp += a*dE
    #print(E / Z)
    return E2

def calc_E4(Eks, aks, beta):
    pi_exp = - calc_lnZ(Eks, aks, beta, rmpf = True)
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    E4 = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        T = A * dE / 2.
        if not (A == mp.mpf(0.)):
            full_exp = 2. * mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.) - 5.* mp.log(np.abs(A))) * 24. * (np.sign(A))
            M = (1.       - (Ek*A)            + ((Ek**2) * (A**2)/2) + (((dE/2)**2)* (A**2)/2) 
                              - ((Ek**3)* (A**3)/6) - (Ek* (A**3.) * ((dE/2)**2)/2)
                              + ((A**4)* (Ek**4)/24) + ((A**4)*(Ek**2)*((dE/2)**2)/4) + ((A**4)*((dE/2)**4)/24)) 
            N = ( - ((dE/2) * A) + ((A**2)*Ek*(dE/2)) 
                            - ((A**3)*(Ek ** 2)*(dE/2)/2) - ((A**3)*((dE/2)**3)/6) 
                            + ((A**4)*(Ek ** 3)*(dE/2)/6) + ((A**4)*((dE/2)**3)*Ek/6)) 
            E4_k = full_exp *(mp.sinh(T)*M + mp.cosh(T)*N)
            E4 += E4_k
            if np.abs(A) < 10**-7 : 
                print(mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.) + mp.log(mp.sinh(T))+ mp.log(abs(M))  + mp.log(mp.cosh(T))  - 5 * mp.log(np.abs(A))) * np.sign(A) * np.sign(M)
                      + mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.) + mp.log(abs(N)) + mp.log(mp.cosh(T))+  mp.log(mp.sinh(T))  - 5 * mp.log(np.abs(A))) * np.sign(A) * np.sign(N))
                print(E4_k)
                return np.NaN
                             #((Ek ** )*((dE/2)**2)/(2 * (A**2.))) 
        else:
            print('A=0, beta =',beta,', a=', a)
            #full_exp = mp.exp(pi_exp - (Ek*a) + a*(dE/2.))
            #E2 += 2 * full_exp * (((Ek**2.) * dE/2.) + ((1./3.) * ((dE/2.) ** 3.)))
        pi_exp += a*dE
    #print(E / Z)
    return E4

def calc_EN(Eks, aks, beta, N):
    pi_exp = - calc_lnZ(Eks, aks, beta, rmpf = True)
    full_exp = mp.mpf(0)
    dE = Eks[1] - Eks[0]
    EN = mp.mpf(0)
    for Ek, a in zip(Eks,aks):
        A = mp.mpf(a + beta)
        if not(A == mp.mpf(0.)):
            if np.abs(A) < 10**-7 : 
                    print('Very small A will get an error')
            full_exp =  np.math.factorial(N) * 2. * mp.exp(pi_exp + (beta* (Ek - (dE/2.) ) ) + (A*dE / 2.))
            T = A * dE / 2.
            for n in range(N + 1):   
                snh_term = mp.mpf(0)
                for i in range(int(np.floor(n/2.)) + 1):
                    snh_term += ((dE / 2.)**(2*i)) * (Ek**(n-2*i)) / (np.math.factorial(2*i) * np.math.factorial(n - 2*i))
                ch_term = mp.mpf(0)
                if(n >0):
                    for i in range(1 ,int(np.ceil(n/2.)) + 1):
                        ch_term += ((dE / 2.)**(2*i - 1)) * (Ek**(n-2*i + 1)) / (np.math.factorial(2*i - 1) * np.math.factorial(n - 2*i + 1))
                EN += (snh_term*mp.sinh(T) + ch_term * mp.cosh(T)) * full_exp * ((-1.)**(N-n)) * (mp.power(A,(n - N - 1))) 
        else:
            print('A=0, beta =',beta,', a=', a)
        pi_exp += a*dE
    return EN

def calc_fxa_lnZ(fxa_df, final_df, beta):
    dE = final_df['dE'][0]
    final_df = final_df.sort_values(by=['Ek'], ignore_index=True)
    Ek = final_df['Ek'].values[0]; a = final_df['a'].values[0];
    S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values; ln_rhok = calc_lnrho(final_df, Ek); 
    VEV_exp = ((a+beta)*S - a*(Ek) + ln_rhok)  -np.log(len(S)) + np.log(dE)
    vmax = VEV_exp.max()
    Z = mp.mpf(0)   
    for Ek, a in zip(final_df['Ek'].values,final_df['a'].values):
        S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values 
        ln_rhok = calc_lnrho(final_df, Ek)
        VEV_exp = ((a+beta)*S - a*(Ek) + ln_rhok)  -np.log(len(S)) + np.log(dE) - vmax
        for ve in VEV_exp:    
            Z += mp.exp(mp.mpf(ve))
    ln_Z = (vmax + float(mp.ln(Z))) 
    return ln_Z

def calc_fxa_obs(fxa_df, final_df, beta, obs, n, lnz):
    dE = final_df['dE'][0]
    final_df = final_df.sort_values(by=['Ek'], ignore_index=True)
    B = 0.  
    for Ek, a in zip(final_df['Ek'].values,final_df['a'].values):
        S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values 
        P = fxa_df[fxa_df['Ek'].values == Ek][obs].values ** n
        ns =  fxa_df[fxa_df['Ek'].values == Ek]['n'].values 
        #plt.hist(S, histtype='step', bins = 100)
        ln_rhok = calc_lnrho(final_df, Ek)
        VEV_exp = (beta*S + a*(S - Ek) + ln_rhok - lnz)
        B += np.mean(P * dE * np.exp(VEV_exp))
        #plt.plot(Ek, np.mean(dE * np.exp(VEV_exp)) , 'kx')
    #plt.show()
    #print(obs,'^',n,':', B)
    return B

def calc_prob_distribution(final_df, beta, lnz, xs = np.array([])):
    if(xs.shape[0] == 0):
        xs = np.linspace(np.min(final_df['Ek'].values + final_df['dE'].values) / (6*final_df['V'].values[0]) , np.max(final_df['Ek'].values) / (6*final_df['V'].values[0]), 1000)
        #print('Here')
    ys = np.zeros(xs.size)
    for x, i in zip(xs,range(len(xs))):
        ys[i] = np.exp(calc_lnrho(final_df, x * (6*final_df['V'].values[0])) + beta*x*(6*final_df['V'].values[0]) - lnz)
    return np.array(xs), np.array(ys)

def prepare_data(LLR_folder, n_repeats, n_replicas, std_files, std_folder, betas, betas_critical, calc_poly = True):
    std_df, hist_df = standard.CSV(std_files, std_folder)
    for nr in range(n_repeats):
        files = [f'{LLR_folder}{nr}/Rep_{j}/out_0' for j in range(n_replicas)]
        RM, fa_df, final_df = CSV(files , f'{LLR_folder}{nr}/CSV/')
        comp_dF = ReadObservables(std_df['Beta'].values,final_df,fa_df,f'{LLR_folder}{nr}/CSV/',file = 'comparison.csv', calc_poly = calc_poly)
        full_dF = ReadObservables(betas,final_df,fa_df,f'{LLR_folder}{nr}/CSV/',file = 'obs.csv', calc_poly = calc_poly)
        critical_dF = ReadObservables(betas_critical,final_df,fa_df,f'{LLR_folder}{nr}/CSV/',file = 'obs_critical.csv', calc_poly=False)

def half_intervals(originalfolder, reducedfolder, mode='even'):
    for i in range(20):
        RM = pd.read_csv(originalfolder + str(i) + '/CSV/RM.csv')
        FA = pd.read_csv(originalfolder + str(i) + '/CSV/fa.csv')
        FNL = pd.read_csv(originalfolder + str(i) + '/CSV/final.csv')
        print(len(np.sort(np.unique(FNL['Ek'].values))[0:-1:2]))
        if mode=='even': Eks = (np.sort(np.unique(FNL['Ek'].values))[0:-1:2])
        elif mode=='odd': Eks = (np.sort(np.unique(FNL['Ek'].values))[1::2])
        elif mode=='mean':  Eks =( (np.sort(np.unique(FNL['Ek'].values))[0:-1:2]) + (np.sort(np.unique(FNL['Ek'].values))[1::2])) /2 
        RM_red = pd.DataFrame(); FA_red = pd.DataFrame(); FNL_red = pd.DataFrame();
        dE = Eks[1] - Eks[0]
        for ek in Eks:    
            RM_red =pd.concat([RM_red, RM[RM['Ek'] == ek]])
            FA_red =pd.concat([FA_red, FA[FA['Ek'] == ek]])
            FNL_red =pd.concat([FNL_red, FNL[FNL['Ek'] == ek]])
        RM_red['dE'] = np.ones_like(RM_red['Ek'])*dE
        FA_red['dE'] = np.ones_like(FA_red['Ek'])*dE
        FNL_red['dE'] = np.ones_like(FNL_red['Ek'])*dE
        RM_red.to_csv(reducedfolder + str(i) + '/CSV/RM.csv', index = False)
        FA_red.to_csv(reducedfolder+ str(i) + '/CSV/fa.csv', index = False)
        FNL_red.to_csv(reducedfolder + str(i) + '/CSV/final.csv', index = False)