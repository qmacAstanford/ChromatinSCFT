import numpy as np
from scipy.special import spherical_jn
from numba import jit
import warnings

@jit(nopython=True)
def M_matrix(U, basis, R=1.0, b=0.05):
    """Matrix for 1D SCFT coefficients

    Args:
        U (ndarray): Array of length n specifying applied potential
        basis (ndarray): shape=(len(xpts),basis_size) of sin(n*pi*x_i/R)
        R (float): Radious or side length (depending on geometery)
        b (float): Kuhn legth

    Return:
        M (numpy array): (basis_size x basis_size) for da/dn = M a
    """
    basis_size = basis.shape[1]
    nxpts = basis.shape[0]
    M = np.zeros((basis_size,basis_size))
    for m in range(1,basis_size+1):
        for n in range(m,basis_size+1):
            # when calculating intigral dx = D/npts
            M[n-1,m-1] = -(2.0/nxpts)*np.sum(basis[:,m-1]*basis[:,n-1]*U)
            M[m-1,n-1] = M[n-1,m-1]
        M[m-1,m-1] = M[m-1,m-1] - ((b*m*np.pi/R)**2)/6
    return M

def calcBasis(xpts,basis_size,R):
    """Caculates basis functions (sin functions) at xpts

    Args:
        xpts (ndarray): Array of length n specifying x coordinates
        basis_size (int): Number of basis functions
        R (float): Radious or side length (depending on geometery)

    Returns:
        len(xpts) by basis_size numpy array of sin(n*pi*x_i/R)
    """
    out = np.zeros((len(xpts),basis_size))
    for n in range(1,basis_size+1):
        out[:,n-1] = np.sin(n*np.pi*xpts/R)
    return out


def density(a, xpts, R=1, fraction=0.5, spherical=False):
    """Reverse Fourier Transform back to position space

    Args:
        a (ndarray): coefficents in forier or bessel series
        xpts (ndarray): x or r positions
        R (float): Radious or side length (depending on geometery)
        fraction (float): mean density to normalize to
        sperical (bool): Use jacobian for spherical coordinates

    Returns:
        phi (ndarray): densities at xpts (same length as xpts)
    """
    basis_size = len(a)
    phi = np.zeros(len(xpts))
    for n in range(1,basis_size+1):
        if spherical:
            if True:
                # note that the coefficients we have solved for differe
                # from the coefficients in front of the bessel function by n
                phi = phi + a[n-1]*n*spherical_jn(0,n*np.pi*xpts/R)
            else:
                phi = phi + a[n-1]*(R/xpts)*np.sin(n*np.pi*xpts/R)

        else:
            phi = phi + a[n-1]*np.sin(n*np.pi*xpts/R)
    phi = phi**2
    phi = normalize(phi, fraction, xpts, R, spherical=spherical)
    return phi

def normalize(phi, fraction, xpts, R, spherical=False):
    # set average volume fraction to fraction
    if spherical:
        f_calc = (3/(R**2))*np.mean(xpts*xpts*phi)
        phi = fraction*phi/f_calc
    else:
        phi = fraction*phi/np.mean(phi)
    return phi

def calc_U(xpts, phi_a, phi_b, rbind=0.1, ubind=0.0, chi_aa=0.0, chi_pp=0.0,
           equ_of_state='FloryHuggins', reverse_x=False, maxX=1.0):
    # TODO: correctly incorporate polymer cross sectional area.
    phi_poly = phi_a + phi_b
    nxpts=len(xpts)
    U_a = np.zeros(nxpts)
    U_b = np.zeros(nxpts)

    if equ_of_state=='FloryHuggins':
        v_over_phi = phi_poly*chi_pp

    U_a = U_a + v_over_phi
    U_b = U_b + v_over_phi

    if (reverse_x):
        u = ubind*(xpts>maxX-rbind)
    else:
        u = ubind*(xpts<rbind)

    U_a = U_a + u
    U_b = U_b + u

    U_a = U_a + chi_aa*phi_a
    return [U_a, U_b]

def energy_terms(phi_a, phi_b, chi_pp, chi_aa, ubind, rbind, xpts, spherical,
                 reverse_x, maxX):
    #  The units of this differ from the units used in the calculation by a
    #  constat.
    out = {}
    nxpts = len(xpts)
    if spherical:
        out['e_chi_pp'] = np.sum(4*np.pi*(xpts**2)*chi_pp*((phi_a+phi_b)**2))\
                        *(maxX/nxpts)
        out['e_chi_aa'] = np.sum(4*np.pi*(xpts**2)*chi_aa*(phi_a**2))\
                        *(maxX/nxpts)
    else:
        out['e_chi_pp'] = np.sum(chi_pp*((phi_a+phi_b)**2))*(maxX/npts)
        out['e_chi_aa'] = np.sum(chi_aa*(phi_a**2))*(maxX/npts)


    if (reverse_x):
        out['e_bind'] = np.sum((phi_a+phi_b)*ubind*(xpts>maxX-rbind))
    else:
        out['e_bind'] = np.sum((phi_a+phi_b)*ubind*(xpts<rbind))

    return out

def is_symmetric(M,tol=1e-8):
    return np.allclose(M, M.T, atol=tol)

def propagator(Lam, L, Q):
    """Calculate
    Q exp(L*Lam) Q.T
    """
    #MM = np.dot(np.diag(np.exp(L*Lam)),Q.T) # slow version
    MM = np.multiply(np.exp(L*Lam)[:,None],Q.T)
    MM = np.dot(Q,MM)
    return MM


@jit
def copolymer_density(M_a, M_b, basis, L_A, L_B, fraction_A, fraction_B, xpts,
                      R, spherical, basis2=None):
    """Calculate A and B density profiles for an alturnating copolymer.

    Args:
        M_a (ndarray): shape=(basis_size, basis_size) differential equation inside A
        M_b (ndarray): shape=(basis_size, basis_size) differential equation inside B
        basis (ndarray): shape=(len(xpts),basis_size) of sin(n*pi*x_i/R)
        L_A (float): length of A block in Kuhn lengths
        L_B (float): length of B block in Kuhn lengths
        fraction_A (float): Valume fraction of A
        fraction_B (float): Valume fraction of B
        nxpts (int): number of descete positions to do calculation at
        R (float): Radious or side length (depending on geometery)
        spherical (bool): In spherical confinment
        basis2 (ndarray): For acceleration

    Return:
        [phi_a, phi_b]
    """
    nxpts = basis.shape[0]
    if not is_symmetric(M_a):
        raise ValueError("M_a must be symmetric")
    if not is_symmetric(M_b):
        raise ValueError("M_b must be symmetric")
    Lam_A, Q_A = np.linalg.eig(M_a)
    Lam_B, Q_B = np.linalg.eig(M_b)

    MMA = propagator(Lam_A, L_A, Q_A)
    MMB = propagator(Lam_B, L_B, Q_B)

    lam, alpha_B = sorted_eig(np.dot(MMB,MMA), top=True)
    if np.abs(np.imag(lam)) > 10**-9:
        raise ValueError("Complex eigenvalue.  Answer shouldn't occolate")
    lam, alpha_A = sorted_eig(np.dot(MMA,MMB), top=True)
    if np.abs(np.imag(lam)) > 10**-9:
        raise ValueError("Complex eigenvalue.  Answer shouldn't occolate")
    if np.max(np.abs(np.imag(alpha_A))) > 10**-8:
        raise ValueError("Complex eigenvector?")
    if np.max(np.abs(np.imag(alpha_B))) > 10**-8:
        raise ValueError("Complex eigenvector?")
    alpha_A = np.real(alpha_A)
    alpha_B = np.real(alpha_B)

    QEQ_A = QEQ(Q_A, alpha_B, Lam_A, L_A)
    QEQ_B = QEQ(Q_B, alpha_A, Lam_B, L_B)

    phi_a = np.zeros(nxpts)
    phi_b = np.zeros(nxpts)
    if spherical:
        if type(basis2) == type(None):
            basis2 = calcBasis2(xpts, basis_size, R)
        #seq = np.arange(basis.shape[1]) + 1
    for ix in range(0,nxpts):
        if spherical:
            #s = basis[ix,:] * (R/xpts[ix])
            #s = seq*spherical_jn(0,seq*np.pi*xpts[ix]/R)
            s = basis2[ix,:]
        else:
            s = basis[ix,:]
        phi_a[ix] = np.linalg.multi_dot([s,QEQ_A,s])
        phi_b[ix] = np.linalg.multi_dot([s,QEQ_B,s])

    phi_a = normalize(phi_a, fraction_A, xpts, R, spherical)
    phi_b = normalize(phi_b, fraction_B, xpts, R, spherical)

    return [phi_a, phi_b]

def calcBasis2(xpts,basis_size,R):
    """Caculates basis functions (sin functions) at xpts
    Args:
        xpts (ndarray): Array of length n specifying x coordinates
        basis_size (int): Number of basis functions
        R (float): Radious or side length (depending on geometery)

    Returns:
        len(xpts) by basis_size numpy array of (R/xi)*sin(n*pi*x_i/R)
    """
    out = np.zeros((len(xpts),basis_size))
    for n in range(1,basis_size+1):
        out[:,n-1] = n*spherical_jn(0,n*np.pi*xpts/R)
        # Alturnatively
        #out[:,n-1] = (R/xpts)*np.sin(n*np.pi*xpts/R)
    return out

@jit(nopython=True)
def QEQ(Q,alpha,Lam,L,tolerence=10**-6):
    basis_size = Q.shape[0]
    out = np.zeros((basis_size,basis_size))
    for nn in range(0,basis_size):
        out[nn,nn] = L*np.exp(L*Lam[nn])
        for mm in range(nn,basis_size):
            if abs((Lam[nn]-Lam[mm])*L) < tolerence:
                x = L*(Lam[nn]-Lam[mm])
                out[nn,mm]=L*np.exp(L*Lam[mm])*(1.0 + 0.5*x + (1.0/6)*x**2)
            else:
                out[nn,mm]=(np.exp(Lam[nn]*L) - np.exp(Lam[mm]*L)) /\
                           (Lam[nn] - Lam[mm])
    QTa = np.dot(Q.T,alpha)
    out = np.multiply(out,QTa)
    #out = np.multiply(QTa[:, np.newaxis],out)
    out = np.multiply(out.T,QTa).T
    #out = np.linalg.multi_dot([Q,out,Q.T])
    out = np.dot(out,Q.T)
    out = np.dot(Q,out)
    return out


def error_metric(phi_1, phi_2, spherical=False, xpts=None):
    """Quntify difference between two density profiles
    Roughly the average fractional deviation between between phi_1 and phi_2.

    Args:
        phi_1 (ndarray): density values
        phi_2 (ndarray): density values to compair to
        spherical (bool): In spherical coordnates
        xpts (ndarray): x possition of density values.  Only used if spherical

    Returns:
        Error metric in range [0,1]
    """
    if spherical:
        return sum(abs(phi_1-phi_2)*(xpts**2))/(2.0*sum(abs(phi_1)*(xpts**2)))
    else:
        return sum(abs(phi_1-phi_2))/(2.0*sum(phi_1))

def sorted_eig(M,top=False,power_iteration=False, guess=None, min_cycles=100,
               tol=10**(-14), max_cycles=10**9):
    if top and power_iteration:
        # choose guess that is unlikely not to work
        if type(guess)==type(None):
            guess=np.ones(M.shape[1])
            guess[0]=2.1
        for ii in range(max_cycles//min_cycles + 1):
            for _ in range(min_cycles):
                b=np.dot(M,guess)
                guess=b/np.linalg.norm(b)
            b=np.dot(M,guess)
            b=b/np.linalg.norm(b)
            ii=np.argmax(abs(b))
            guess=guess*b[ii]/guess[ii]
            if np.amax(abs(b-guess)) < tol:
                return b
            guess=b
        return ValueError("Failed to convirge in ", max_cycles, "cycles")
    # Otherwise use eigenvalue methode
    eig_vals, eig_vecs = np.linalg.eig(M)
    if top:
        ii = np.argmax(eig_vals)
        return [eig_vals[ii],eig_vecs[:,ii]]
    sort_perm = eig_vals.argsort()
    eig_vals.sort()
    eig_vecs = eig_vecs[:, sort_perm]
    return [eig_vals, eig_vecs]

def progression(params_0, delta_params, n_samples, save_history=False):
    parameters = {}
    u_constants = {}
    u_args ={'rbind', 'ubind', 'chi_aa', 'chi_pp', 'equ_of_state', 'reverse_x',
             'maxX'}
    iterate_args = {'tolerence', 'cycles', 'R', 'b', 'basis_size', 'nxpts',
            'fraction_A', 'fraction_B', 'fractional_update',
            'accelerated_update', 'accelerate_level',
            'spherical', 'init_phi_a', 'init_phi_b', 'L_A', 'L_B'}
    for key in params_0:
        if key in u_args:
            u_constants[key] = params_0[key]
        if key in iterate_args:
            parameters[key] = params_0[key]
        if key not in u_args and key not in iterate_args:
            raise ValueError(str(key)+" is not a keyward")

    parameters['flagFail']=True
    samples = []
    for i_sample in range(n_samples):
        print("Takeing sameple ",i_sample," ...")
        for key in delta_params:
            if key in u_args:
                u_constants[key] = params_0[key] + i_sample*delta_params[key]
            if key in iterate_args:
                parameters[key] = params_0[key] + i_sample*delta_paras[key]

        [xpts,history,flagFail]=iterate(u_constants,**parameters)

        errors = []
        for ii in range(len(history)):
            errors.append(history[ii]['error'])

        # Save data about sample
        samples.append({'phi_a':history[-1]['phi_a'],
                        'phi_b':history[-1]['phi_b'],
                        'errors':errors
                        })
        if save_history:
            samples[-1]['history'] = history
        samples[-1]['failToConverg'] = flagFail

        # Save the changing arguement(s)
        for key in delta_params:
            if key in u_args:
                samples[-1][key] = u_constants[key]
            elif key in iterate_args:
                samples[-1][key] = parameters[key]

        # use final densities for next initial condition
        parameters['init_phi_a']=history[-1]['phi_a']
        parameters['init_phi_b']=history[-1]['phi_b']

    return [xpts, samples]

def run_iterate_once(params_0):
    parameters = {}
    u_constants = {}
    u_args ={'rbind', 'ubind', 'chi_aa', 'chi_pp', 'equ_of_state', 'reverse_x',
             'maxX'}
    iterate_args = {'tolerence', 'cycles', 'R', 'b', 'basis_size', 'nxpts',
            'fraction_A', 'fraction_B', 'fractional_update',
            'accelerated_update', 'accelerate_level',
            'spherical', 'init_phi_a', 'init_phi_b', 'L_A', 'L_B'}
    for key in params_0:
        if key in u_args:
            u_constants[key] = params_0[key]
        if key in iterate_args:
            parameters[key] = params_0[key]
        if key not in u_args and key not in iterate_args:
            raise ValueError(str(key)+" is not a keyward")

    return iterate(u_constants,**parameters)


def iterate(u_constants, tolerence=0.005, cycles=100, R=1.0, b=0.05,
            basis_size=150, nxpts=450,
            fraction_A=0.01, fraction_B=0.01, fractional_update=0.05,
            accelerated_update=0.05, accelerate_level=0.001,
            spherical=False, init_phi_a=None, init_phi_b=None, flagFail=False,
            L_A=None, L_B=None):
    """Run SCFT proceedure

    Args:
        u_constants (dict): passed to calc_U
        tolerence (float): acceptable deviation in error metric
        cycles (int): maximum number of iterations
        R (float): Radious or side length (depending on geometery)
        b (float): Kuhn length
        basis_size (int): Number of basis functions
        nxpts (int): number of descete positions to do calculation at
        fraction_A (float): Valume fraction of A
        fraction_B (float): Valume fraction of B
        fractional_update (float): how much to updat on each iteration choose from range (0,1]
        accelerated_update (float): fractional_update when error below accelerate_level
        accelerate_level (float): error level at which to increase fractional_update
        spherical (bool): In spherical confinment
        init_phi_a (ndarray): shape=(nxpts) initial density of A
        init_phi_b (ndarray): shape=(nxpts) initial density of B
        flagFail (bool): also return True/False if tolerence is reached
        L_A (float): length of A block
        L_B (float): length of B block

    Returns:
        [xpts, history]
        xptx is the numpy array of x positions
        history is of list of length cycles of dictionaries containing data

        Example: history[-1]['phi_a] is the final phi_a density profile
    """
    u_constants['maxX'] = R
    if u_constants['reverse_x']==None:
        if spherical:
            u_constants['reverse_x'] = True
        else:
            u_constants['reverse_x'] = False

    xpts = np.linspace(0.0,R,nxpts+2)[1:-1]

    #initial condition
    if type(init_phi_a) == type(None):
        phi_a = fraction_A*np.ones(nxpts)
    else:
        phi_a = normalize(init_phi_a, fraction_A, xpts, R, spherical=spherical)
    if type(init_phi_b) == type(None):
        phi_b = fraction_B*np.ones(nxpts)
    else:
        phi_b = normalize(init_phi_b, fraction_B, xpts, R, spherical=spherical)

    # input check for alturnating
    if (type(L_A) != type(L_B) ):
        print("L_A ",L_A)
        print("L_B ",L_B)
        raise ValueError("If supplied, both L_A and L_B must be floats.")
    if (type(L_A) != type(None)):
        is_copolymer = True
    else:
        is_copolymer = False

    history = []

    basis = calcBasis(xpts, basis_size, R)
    if is_copolymer and spherical:
        basis2 = calcBasis2(xpts, basis_size, R)
    else:
        basis2 = None

    for ii in range(cycles):
        [U_a, U_b] = calc_U(xpts, phi_a, phi_b, **u_constants)

        M_a = M_matrix(U_a, basis, R=R, b=b)
        M_b = M_matrix(U_b, basis, R=R, b=b)

        if is_copolymer:
            phi_a_new, phi_b_new = copolymer_density(M_a, M_b, basis, L_A,
                                                     L_B, fraction_A,
                                                     fraction_B, xpts, R,
                                                     spherical, basis2)
        else:
            [lam_a, a_a] = sorted_eig(M_a,top=True)
            [lam_b, a_b] = sorted_eig(M_b,top=True)

            phi_a_new = density(a_a, xpts, R=R, fraction=fraction_A,
                                spherical=spherical)
            phi_b_new = density(a_b, xpts, R=R, fraction=fraction_B,
                                spherical=spherical)

        error = 0.5*(error_metric(phi_a,phi_a_new, spherical=spherical,
                                  xpts=xpts) +
                     error_metric(phi_b,phi_b_new, spherical=spherical,
                                  xpts=xpts))

        if error<accelerate_level:
            phi_a = phi_a*(1.0-accelerated_update) + accelerated_update*phi_a_new
            phi_b = phi_b*(1.0-accelerated_update) + accelerated_update*phi_b_new
        else:
            phi_a = phi_a*(1.0-fractional_update) + fractional_update*phi_a_new
            phi_b = phi_b*(1.0-fractional_update) + fractional_update*phi_b_new

        #history.append({'phi_a':phi_a, 'phi_b':phi_b, 'lam_a':lam_a,
        #                'lam_b':lam_b, 'U_a':U_a, 'U_b':U_b, 'M_a':M_a,
        #                '`M_b':M_b, 'a_a':a_a, 'a_b':a_b,
        #                'error':error, 'phi_a_proposed':phi_a_new,
        #                'phi_b_porposed':phi_b_new})
        history.append({'phi_a':phi_a, 'phi_b':phi_b, 'U_a':U_a, 'U_b':U_b,
                        'error':error, 'phi_a_proposed':phi_a_new,
                        'phi_b_porposed':phi_b_new})

        if error < tolerence:
            if flagFail:
                return[xpts, history, False]
            else:
                return[xpts, history]
    if flagFail:
        return [xpts, history, True]

    if tolerence != None:
        warnings.warn("Failed to converge error metric = "+str(error))
    return [xpts, history]


