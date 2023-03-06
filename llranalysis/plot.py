import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import llranalysis.llr as llr
import llranalysis.error as error
import llranalysis.standard as standard
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  


plt.style.use('default')
plt.rcParams.update({'xtick.labelsize' : 18,
                     'ytick.labelsize' : 18,
                     'axes.formatter.useoffset' : False,
                     'legend.fontsize' : 20,
                     'axes.labelsize' : 30,
                     "text.usetex": True,
                     "font.family": "serif",
                     "font.serif": "Computer Modern Roman",
                     'lines.linewidth':2,
                     'figure.figsize' : (10,10),
                     'figure.autolayout': True})

def plot_RM_repeats(boot_folder, n_boots, interval):
    RM_df_names = [boot_folder + str(m) + '/CSV/RM.csv' for m in range(n_boots)]
    figure_folder = boot_folder + 'Figures/'
    RM = pd.read_csv(RM_df_names[0])
    Eks = np.unique(RM['Ek'])
    fig, ax =plt.subplots(figsize=(10,10)) # 
    lst_a = []
    for RM_df_name in RM_df_names:
        RM = pd.read_csv(RM_df_name)
        Ek = Eks[interval]
        plt.plot(-RM[RM['Ek'] == Ek]['a'].values) # axs[0]
        plt.xlabel('RM iteration m') #axs[0].set_
        plt.ylabel('$a_n^{(m)}$') #axs[0].set_
        lst_a.append(-RM[RM['Ek'] == Ek]['a'].values[-1])
    
    axsins = inset_axes(ax, width = 3, height = 3, loc='upper left',
                   bbox_to_anchor=(0.5,1-0.4,.3,.3), bbox_transform=ax.transAxes)
    axsins.hist(lst_a, orientation="horizontal",histtype='step')
    axsins.set_ylabel('$a_n$')
    axsins.set_xticks([])
    axsins.tick_params(axis='both', which='major', labelsize=15)
    axsins.tick_params(axis='both', which='minor', labelsize=15)
    axsins.locator_params(axis="y", nbins=5)
    plt.show()

def plot_RM_swaps(boot_folder, repeat):
    RM_df = pd.read_csv(boot_folder + str(repeat) + '/CSV/RM.csv')
    RM_swap_df = RM_df.sort_values(by=['Rep','n'], ignore_index = True)
    for r in zip(np.unique(RM_swap_df['Rep'])):
        plt.plot(-RM_swap_df[RM_swap_df['Rep'] == r]['a'].values)
        plt.xlabel('RM iteration m') # ,fontsize = 30
        plt.ylabel('$a_n^{(m)}$') # , fontsize = 30
    plt.ylim([-RM_df['a'].max(), -RM_df['a'].min()])
    plt.show()

def plot_comparison_histograms(boot_folder, n_repeats, std_files, std_folder,num_samples = 200, error_type= 'standard deviation'):
    std_df, hist_df = standard.CSV(std_files, std_folder)
    colours = ['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
    for i, beta in enumerate(std_df['Beta'].values):
        xs = np.array([]);ys = np.array([])
        for nr in range(n_repeats):
            final_df = pd.read_csv(f'{boot_folder}{nr}/CSV/final.csv')
            V = final_df['V'].values[0]
            lnz = float(llr.calc_lnZ(final_df['Ek'].values, final_df['a'].values, beta))
            x, y = llr.calc_prob_distribution(final_df, beta, lnz)
            xs = np.append(xs, x); ys = np.append(ys, y)
        xs.shape = [n_repeats, len(x)]; ys.shape = [n_repeats, len(y)]
        xs = xs.mean(axis = 0)
        ys_err = error.calculate_error_set(ys, num_samples, error_type)
        ys = ys.mean(axis = 0)
        plt.plot(xs,(ys + ys_err)* (6*V), 'b--')
        plt.plot(xs,ys* (6*V), 'b-')
        plt.plot(xs,(ys - ys_err)* (6*V), 'b--')
        hist_tmp = hist_df[hist_df['Beta'] == beta]['Hist'].values
        bins_tmp = hist_df[hist_df['Beta'] == beta]['Bins'].values
        plt.plot(bins_tmp,hist_tmp, c = 'orange', ls = '-') #, lw = 1
    plt.yticks([])
    plt.plot(np.NaN, np.NaN, 'b-', label='LLR')
    plt.plot(np.NaN, np.NaN, c='orange', ls='-', label='Importance sampling')
    plt.legend() 
    plt.ylabel('$P_{\\beta}(u_p)$' )
    plt.xlabel('$u_p$')
    plt.show()


def fxa_hist(boot_folder, selected_repeat, ulim):
    _, fxa_df, final_df = llr.ReadCSVFull(f'{boot_folder}{selected_repeat}/CSV/')
    V = final_df['V'].values[0]   
    for Ek, a in zip(final_df['Ek'].values,final_df['a'].values):
        S = fxa_df[fxa_df['Ek'].values == Ek]['S'].values 
        S = S[S != 0]
        plt.hist(S / (6*V), histtype='step', bins = 20, density = True)
    plt.xlabel('$u_p$', fontsize = 30)
    plt.ylabel('$P_{\\beta}(u_p)$', fontsize = 30)
    #plt.xlim(ulim)
    #plt.title('dE ='+ str(final_df['dE'].values[0]))
    plt.yticks([],[])
    plt.show()