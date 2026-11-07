import numpy as np
from scipy import special as scyspecial
import scipy.signal as scysignal

import utils

matlab_dir = 'srir/ambisonics_dependencies'
reg_type_dict = {
    'tikhonov': 'tikhonov',
    'open': 'soft',
    'rigid': 'hard'
    }

class Ambisonics():
    """Class to encode a signal from a format to ambisonics format.

    """    

    def __init__(self, SH_order=1, azi=None, ele=None, fs=24000, n_points=2048,
                 radius=0.042, c=343, array_type='open', SH_type='real'):
        """
        Parameters
        ----------
        SH_order : int, optional
            The order of the spherical harmonics, by default 1
        azi : (num_mic,) array_like, optional
            azimuth angles of the microphones array in degrees, by default None
        ele : (num_mic,) array_like, optional
            elevation angle of the microphone array in degrees, by default None
        array_type : str, {'rigid', 'open'}, optional
            The type of micophone array, by default 'open'

        """        
        if array_type not in ['open', 'rigid']:
            raise ValueError('Array type must be \'open\' or \'rigid\'.')
        
        if (azi is None) ^ (ele is None):
            raise ValueError('azi and ele must be both specified or None.')

        if SH_order == 1 and azi is None and ele is None:
            mic_pos = np.array([[45, 35], [-45, -35], 
                                [135, -35], [-135, 35]])
            azi = mic_pos[:, 0]
            ele = mic_pos[:, 1]
        else:
            azi = np.array(azi)
            ele = np.array(ele)
        
        self.SH_order = SH_order
        self.SH_type = SH_type
        self.azi = azi / 180 * np.pi
        self.ele = ele / 180 * np.pi
        self.colat = np.pi / 2 - self.ele
        self.array_type = array_type
        self.fs = fs

        f = np.linspace(0, fs//2, n_points//2+1)
        k = 2 * np.pi * f / c
        self.SH_matrix = self.sh_matrix(
            self.SH_order, self.azi, self.colat, SH_type=SH_type)
        
        try:
            self.bn = self.get_sma_radial_filters(
                k=k, reg_type=reg_type_dict[array_type], 
                r=radius, matlab_dir=matlab_dir)
        except:
            self.bn = None


    def get_sma_radial_filters(self, k, reg_type='tikhonov', r=0.042,
                               matlab_dir='./ambisonics_dependencies'):
        # REF: https://github.com/AppliedAcousticsChalmers/ambisonic-encoding
        import matlab
        import matlab.engine

        eng = matlab.engine.start_matlab()
        eng.cd(matlab_dir)

        b_n, b_n_inv, b_n_inv_t = eng.get_sma_radial_filters(
            matlab.double(k[:,None].tolist()), 
            float(r), float(self.SH_order), 20.0, reg_type, 2, 
            nargout=3)
        b_n = np.array(b_n).T
        b_n_inv = np.array(b_n_inv).T
        b_n_inv_t = np.array(b_n_inv_t).T
        return b_n, b_n_inv, b_n_inv_t


    def encoding(self, signal, norm='SN3D'):
        """Encode a audio signal to ambisonics format.

        Parameters
        ----------
        signal : (num_mic, num_samples), array_like
            The input audio signal.
        SH_type : str, {'real', 'complex'}
            The type of spherical harmonics function
        c : float
            Speed of sound in m/s
        norm : str, {'N3D', 'SN3D'}, optional
            Normalization of the SH basis, by default 'SN3D'
        radius : float, optional
            Radius of the sphere in meters, by default 0.042
        fs : int, optional
            Sampling rate, by default 24000

        Returns
        -------
        Signal_hoa: (num_mic, num_samples), array_like
            The encoded signal.
        """        
        if self.SH_type not in ['real', 'complex']:
            raise ValueError('SH_type must be \'real\' or \'complex\'.')       
        
        b_n, b_n_inv, b_n_inv_t = self.bn

        if self.SH_type == 'real':
            # The convention used here is also known as N3D-ACN (for SH_type='real').
            signal = self.SH_matrix.T @ signal
        elif self.SH_type == 'complex':
            raise NotImplementedError('Complex SH_type is not implemented yet.')
        b_n_inv_t = np.repeat(b_n_inv_t, 2*np.arange(self.SH_order+1)+1, axis=0)
        signal_hoa = scysignal.fftconvolve(b_n_inv_t, signal, axes=-1)

        if norm == 'SN3D':
            return self.N3D_to_SN3D(signal_hoa)
        elif norm == 'N3D':
            return signal_hoa
        else:
            raise ValueError('Normalization type {} not supported'.format(norm))

    
    def N3D_to_SN3D(self, F_nm, sh_axis=0):
        """Convert N3D (orthonormal) to SN3D (Schmidt semi-normalized) signals.

        Parameters
        ----------
        F_nm : ((N_sph+1)**2, S) numpy.ndarray
            Matrix of spherical harmonics coefficients of spherical function(S).
        sh_axis : int, optional
            SH axis. The default is 0.

        Returns
        -------
        F_nm : ((N_sph+1)**2, S) numpy.ndarray
            Matrix of spherical harmonics coefficients of spherical function(S).

        """
        assert(F_nm.ndim == 2)
        # Input SH order
        N = int(np.sqrt(F_nm.shape[sh_axis]) - 1)
        # 1/sqrt(2n+1) conversion factor
        n_norm = np.array([1/np.sqrt(2*n + 1) for n in range(N + 1)])
        # Broadcast
        n_norm = np.expand_dims(utils.repeat_per_order(n_norm), axis=sh_axis-1)
        return n_norm * F_nm


    @staticmethod
    def mode_strength(n, kr, sphere_type='rigid'):
        """Mode strength b_n(kr) for an incident plane wave on sphere.

        REF: https://github.com/chris-hld/spaudiopy

        Parameters
        ----------
        n : int
            Degree.
        kr : array_like
            kr vector, product of wavenumber k and radius r_0.
        sphere_type : 'rigid' or 'open'

        Returns
        -------
        b_n : array_like
            Mode strength b_n(kr).

        References
        ----------
        Rafaely, B. (2015). Fundamentals of Spherical Array Processing. Springer.
        eq. (4.4) and (4.5).
        """

        np.seterr(divide='ignore', invalid='ignore')

        def spherical_hn2(n, z, derivative=False):
            """Spherical Hankel function of the second kind.

            Parameters
            ----------
            n : int, array_like
                Order of the spherical Hankel function (n >= 0).
            z : complex or float, array_like
                Argument of the spherical Hankel function.
            derivative : bool, optional
                If True, the value of the derivative (rather than the function
                itself) is returned.

            Returns
            -------
            hn2 : array_like


            References
            ----------
            http://mathworld.wolfram.com/SphericalHankelFunctionoftheSecondKind.html
            """
            with np.errstate(invalid='ignore'):
                yi = 1j * scyspecial.spherical_yn(n, z, derivative)
            return scyspecial.spherical_jn(n, z, derivative) - yi

        kr = np.asarray(kr)
        if sphere_type == 'open':
            b_n = 4*np.pi*1j**n * scyspecial.spherical_jn(n, kr)
        elif sphere_type == 'rigid':
            b_n = 4*np.pi*1j**n * (scyspecial.spherical_jn(n, kr) -
                                (scyspecial.spherical_jn(n, kr, True) /
                                spherical_hn2(n, kr, True)) *
                                spherical_hn2(n, kr))
        else:
            raise ValueError('sphere_type Not implemented.')
        
        idx_kr0 = np.where(kr==0)[0]
        idx_nan = np.where(np.isnan(b_n))[0]
        b_n[idx_nan] = 0
        if n == 0:
            b_n[idx_kr0] = 4*np.pi
        else:
            b_n[idx_kr0] = 0

        return b_n


    def sh_matrix(self, N, azi, colat, SH_type='real', weights=None):
        r"""Matrix of spherical harmonics up to order N for given angles.

        Computes a matrix of spherical harmonics up to order :math:`N`
        for the given angles/grid.

        REF: https://github.com/chris-hld/spaudiopy

        .. math::

            \mathbf{Y} = \left[ \begin{array}{ccccc}
            Y_0^0(\theta[0], \phi[0]) & Y_1^{-1}(\theta[0], \phi[0]) &
            Y_1^0(\theta[0], \phi[0]) &
            \dots & Y_N^N(\theta[0], \phi[0])  \\
            Y_0^0(\theta[1], \phi[1]) & Y_1^{-1}(\theta[1], \phi[1]) &
            Y_1^0(\theta[1], \phi[1]) &
            \dots & Y_N^N(\theta[1], \phi[1])  \\
            \vdots & \vdots & \vdots & \vdots & \vdots \\
            Y_0^0(\theta[Q-1], \phi[Q-1]) & Y_1^{-1}(\theta[Q-1], \phi[Q-1]) &
            Y_1^0(\theta[Q-1], \phi[Q-1]) &
            \dots & Y_N^N(\theta[Q-1], \phi[Q-1])
            \end{array} \right]

        where

        .. math::

            Y_n^m(\theta, \phi) = \sqrt{\frac{2n + 1}{4 \pi}
                                        \frac{(n-m)!}{(n+m)!}} P_n^m(\cos \theta)
                                e^{i m \phi}

        When using `SH_type='real'`, the real spherical harmonics
        :math:`Y_{n,m}(\theta, \phi)` are implemented as a relation to
        :math:`Y_n^m(\theta, \phi)`.

        Parameters
        ----------
        N : int
            Maximum SH order.
        azi : (Q,) array_like
            Azimuth.
        colat : (Q,) array_like
            Colatitude.
        SH_type :  'complex' or 'real' spherical harmonics.
        weights : (Q,) array_like, optional
            Quadrature weights.

        Returns
        -------
        Ymn : (Q, (N+1)**2) numpy.ndarray
            Matrix of spherical harmonics.

        Notes
        -----
        The convention used here is also known as N3D-ACN.

        """
        azi = utils.asarray_1d(azi)
        colat = utils.asarray_1d(colat)
        if azi.ndim == 0:
            Q = 1
        else:
            Q = len(azi)
        if weights is None:
            weights = np.ones(Q)
        if SH_type == 'complex':
            Ymn = np.zeros([Q, (N+1)**2], dtype=np.complex128)
        elif SH_type == 'real':
            Ymn = np.zeros([Q, (N+1)**2], dtype=np.float64)
        else:
            raise ValueError('SH_type unknown.')

        idx = 0
        for n in range(N+1):
            for m in range(-n, n+1):
                if SH_type == 'complex':
                    Ymn[:, idx] = weights * scyspecial.sph_harm(m, n, azi, colat)
                elif SH_type == 'real':
                    if m == 0:
                        Ymn[:, idx] = weights * np.real(
                                    scyspecial.sph_harm(0, n, azi, colat))
                    if m < 0:
                        Ymn[:, idx] = weights * np.sqrt(2) * (-1) ** abs(m) * \
                                    np.imag(
                                    scyspecial.sph_harm(abs(m), n, azi, colat))
                    if m > 0:
                        Ymn[:, idx] = weights * np.sqrt(2) * (-1) ** abs(m) * \
                                    np.real(
                                    scyspecial.sph_harm(abs(m), n, azi, colat))

                idx += 1
        return Ymn

    