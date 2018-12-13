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
    M = np.zeros((basis_size,basis_size))
    for m in range(1,basis_size+1):
        for n in range(m,basis_size+1):
            M[n-1,m-1] = -2.0*np.sum(basis[:,m-1]*basis[:,n-1]*U)
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
            phi = phi + a[n-1]*spherical_jn(0,n*np.pi*xpts/R)
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

def progression(params_0,delta_params,n_samples):
    parameters = {}
    u_constants = {}
    u_args ={'rbind', 'ubind', 'chi_aa', 'chi_pp', 'equ_of_state', 'reverse_x',
             'maxX'}
    iterate_args = {'tolerence', 'cycles', 'R', 'b', 'basis_size', 'nxpts',
            'fraction_A', 'fraction_B', 'fractional_update',
            'accelerated_update', 'accelerate_level',
            'spherical', 'init_phi_a', 'init_phi_b'}
    for key in params_0:
        if key in u_args:
            u_constants[key] = params_0[key]
        if key in iterate_args:
            parameters[key] = params_0[key]
        if key not in u_args and key not in iterate_args:
            raise ValueError(str(key)+" is not a keyward")

    samples = []
    for i_sample in range(n_samples):
        print("Takeing sameple ",i_sample," ...")
        for key in delta_params:
            if key in u_args:
                u_constants[key] = params_0[key] + i_sample*delta_params[key]
            if key in iterate_args:
                parameters[key] = params_0[key] + i_sample*delta_paras[key]

        [xpts,history]=iterate(u_constants,**parameters)

        errors = []
        for ii in range(len(history)):
            errors.append(history[ii]['error'])

        # Save data about sample
        samples.append({'phi_a':history[-1]['phi_a'],
                        'phi_b':history[-1]['phi_b'],
                        'errors':errors
                        })
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




def iterate(u_constants, tolerence=0.005, cycles=100, R=1.0, b=0.05,
            basis_size=150, nxpts=450,
            fraction_A=0.01, fraction_B=0.01, fractional_update=0.05,
            accelerated_update=0.05, accelerate_level=0.001,
            spherical=False, init_phi_a=None, init_phi_b=None):
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
        spherical (bool): In spherical confinme
        init_phi_a (ndarray): shape=(nxpts) initial density of A
        init_phi_b (ndarray): shape=(nxpts) initial density of B

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

    history = []

    basis = calcBasis(xpts, basis_size, R)

    for ii in range(cycles):
        [U_a, U_b] = calc_U(xpts, phi_a, phi_b, **u_constants)

        M_a = M_matrix(U_a, basis, R=R, b=b)
        M_b = M_matrix(U_b, basis, R=R, b=b)

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

        history.append({'phi_a':phi_a, 'phi_b':phi_b, 'lam_a':lam_a,
                        'lam_b':lam_b, 'U_a':U_a, 'U_b':U_b, 'M_a':M_a,
                        '`M_b':M_b, 'a_a':a_a, 'a_b':a_b,
                        'error':error, 'phi_a_proposed':phi_a_new,
                        'phi_b_porposed':phi_b_new})
        if error < tolerence:
            return[xpts,history]

    if tolerence != None:
        warnings.warn("Failed to converge error metric = "+str(error))

    return [xpts, history]


