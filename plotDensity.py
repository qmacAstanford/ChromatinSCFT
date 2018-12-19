import oneDInteraction as od
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

font={'family':'serif','weight':'normal','size':18}
plt.rc('font',**font)

def plotDenistyHistory(xpts, history, skip=1):
    colors = sns.cubehelix_palette(len(history))
    colors_b = sns.cubehelix_palette(len(history),start=2,rot=0)
    for m in range(0,len(history),skip):
        plt.plot(xpts,history[m]['phi_a'],color=colors[m])
        plt.plot(xpts,history[m]['phi_b'],color=colors_b[m])
    plt.xlabel('r/R')
    plt.ylabel('VolumeFraction')
    plt.tight_layout()
    plt.show()

def plotPhiAPhiB(xpts,phi_a,phi_b):
    plt.plot(xpts,phi_a,label='a')
    plt.plot(xpts,phi_b,label='b')
    plt.plot(xpts,phi_a+phi_b,label='total',color='k')
    plt.ylabel('Volume Fraction')
    plt.xlabel('r/R')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotProposed(xpts, history, skip=3,minimum=0, maximum=1000):
    npts=min(len(history),maximum)-minimum
    colors = sns.cubehelix_palette(npts)
    colors_b = sns.cubehelix_palette(npts,start=2,rot=0)
    for m in range(minimum,min(len(history),maximum),skip):
        plt.plot(xpts,history[m]['phi_a'],color=colors[m-minimum])
        plt.plot(xpts,history[m]['phi_b'],color=colors_b[m-minimum])
        plt.plot(xpts,history[m]['phi_a_proposed'],color='r')
        plt.plot(xpts,history[m]['phi_b_porposed'],color='b')
    if params_0['spherical']:
        plt.xlabel('r')
    else:
        plt.xlabel('x')
    plt.ylabel('Volume Fraction')
    plt.tight_layout()
    plt.show()


def plot_phase_progression(params_0, delta_params, samples, xpts, setType):
    """Plot to see where phase transition occure
    """
    #----------------
    #  Unpack energies from samples
    #----------------
    what_changed = list(delta_params.keys())[0]
    e_chi_pp = []
    e_chi_aa = []
    e_bind = []
    fail_to_converg = []
    xvals = []
    plotparams = params_0.copy()
    for idx, sample in enumerate(samples):
        for key in delta_params:
            plotparams[key] = params_0[key]+idx*delta_params[key]
        energy = od.energy_terms(sample['phi_a'], sample['phi_b'],
                                 plotparams['chi_pp'], plotparams['chi_aa'],
                                 plotparams['ubind'], plotparams['rbind'],
                                 xpts, plotparams['spherical'], plotparams['reverse_x'],
                                 plotparams['R'])
        e_chi_pp.append(energy['chi_pp'])
        e_chi_aa.append(energy['chi_aa'])
        e_bind.append(energy['bind'])
        fail_to_converg.append(sample['failToConverg'])
        xvals.append(sample[what_changed])
    e_chi_pp = np.array(e_chi_pp)
    e_chi_aa = np.array(e_chi_aa)
    e_bind = np.array(e_bind)
    fail_to_converg = np.array(fail_to_converg)

    #-------------
    # Make plot
    #-------------
    for converge in [True, False]:
        if converge:
            marker = 'o'
            linestyle = '-'
            choice = fail_to_converg == False
        else:
            marker = 'x'
            linestyle = ' '
            choice = fail_to_converg == True
    
            
        x = xvals
        x_vals = []
        for ii, val in enumerate(x):
            if fail_to_converg[ii] == converge:
                x_vals.append(None)
            else:
                x_vals.append(val)
        plt.plot(x_vals, e_chi_pp, marker=marker, linestyle=linestyle,
                 color='b', label='e_chi_pp')
        plt.plot(x_vals, e_chi_aa, marker=marker, linestyle=linestyle,
                 color='r', label='e_chi_aa')
        plt.plot(x_vals, e_bind,  marker=marker, linestyle=linestyle,
                 color='g', label='e_bind')
        plt.xlabel(what_changed)
        plt.title(setType)
        plt.legend()
    plt.tight_layout()

def plot_a_density_samples(samples,xpts):
    colors = sns.cubehelix_palette(len(samples))
    for i_sample, sample in enumerate(samples):
        if sample['failToConverg']:
            linestyle=':'
        else:
            linestyle='-'
        plt.plot(xpts, sample['phi_a'], color=colors[i_sample], linestyle=linestyle)
        plt.title('phi_a')
    if params_0['spherical']:
        plt.xlabel('r')
    else:
        plt.xlabel('x')
    #plt.ylim((0,0.1))
    plt.ylabel('Volume Fraction')
    plt.tight_layout()
    plt.show()

def plot_b_density_samples(samples,xpts):
    colors_b = sns.cubehelix_palette(len(samples),start=2,rot=0)
    for i_sample, sample in enumerate(samples):
        if sample['failToConverg']:
            linestyle=':'
        else:
            linestyle='-'
        plt.plot(xpts,sample['phi_b'],color=colors_b[i_sample], linestyle=linestyle)
        plt.title('phi_b')
    if params_0['spherical']:
        plt.xlabel('r')
    else:
        plt.xlabel('x')
    #plt.ylim((0,0.1))
    plt.ylabel('Volume Fraction')
    plt.tight_layout()
    plt.show()
