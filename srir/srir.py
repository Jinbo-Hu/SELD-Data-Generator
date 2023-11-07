import json
import os, warnings

import numpy as np
import scipy.signal as scysignal
from itertools import product

import utils


class GenerateSRIR():
    """Spatial Room Impulse Response (SRIR) Generation"""
        
    def __init__(
        self, fs=24000., src_dir='omni', mic_dir='omni', radius=0.042, 
        c=343., mic_pos=None, tools='pyroomacoustics'):
        """
        Parameters
        ----------
        fs : int, optional
            Sampling rate, by default 24000
        src_dir : str, optional
            Directivity of sources, by default 'omni'
        mic_dir : str, optional
            Directivity of microphones, by default 'omni'
        radius : float, optional
            Radius of spherical array, by default 0.042
        mic_pos : _type_, optional
            Spherical coordinate position of microphones in degree, 
            by default None

        """        

        self.fs = fs
        self.src_dir = src_dir
        self.mic_dir = mic_dir
        self.radius = radius
        self.c = c

        assert tools in ['pyroomacoustics', 'gpuRIR', 'smir'], \
            "tools must be one of 'pyroomacoustics', 'gpuRIR', 'smir'"
        self.tools = tools
        self.rir = None # (num_mic, num_src, num_points)

        if mic_pos is None:
            raise ValueError('mic_pos must be given.')
        else:
            self.mic_pos_cart = utils.sph2cart(mic_pos[:,0], mic_pos[:,1], radius)
            self.mic_pos_sph = mic_pos


    def compute_srir(
        self, room_dim, src_pos, rt60, 
        mic_pos_center=None, method='ism', 
        **kwargs):
        if self.tools == 'pyroomacoustics':
            self.compute_srir_pra(room_dim, src_pos, rt60, mic_pos_center, method, **kwargs)
        elif self.tools == 'gpuRIR':
            self.compute_srir_gpuRIR(room_dim, src_pos, rt60, mic_pos_center, **kwargs)
        elif self.tools == 'smir':
            self.compute_srir_smir(room_dim, src_pos, rt60, mic_pos_center, **kwargs)
     

    def compute_srir_pra(
        self, room_dim, src_pos, rt60=None, 
        mic_pos_center=None, method='hybrid', 
        **kwargs):
        """Compute an SRIR from a given room parameters
            using pyroomacoustics.

        Parameters
        ----------
        room_dim : (3,) array_like
            Room dimensions in meters
        src_pos : (num_src, 3) array_like
            Source positions in meters
        rt60 : float,
            Desired RT60 in seconds
        mic_pos_center : (3,) array_like, optional
            Position of center of microphone array, 
            The default None, which means the center of room.
        method : str, {'ism', 'hybrid'}, optional
            Method of rir generator, by default 'hybrid', which
            means image source method and ray tracing are used.
        kwargs : dict
            Additional arguments for pyroomacoustics.ShoeBox

        Returns
        -------
        array_like, shape (num_src, num_mic, length)
            Generated SRIR.
        """        

        import pyroomacoustics as pra

        if mic_pos_center is None:
            mic_pos = self.mic_pos_cart.T + np.c_[room_dim]/2
        else:
            mic_pos = self.mic_pos_cart.T + np.c_[mic_pos_center]

        if rt60 is not None:
            # We invert Sabine's formula to obtain the parameters for the ISM simulator
            e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
            if method == "ism":
                room = pra.ShoeBox(
                    p=room_dim, 
                    fs=self.fs, 
                    materials=pra.Material(e_absorption), 
                    max_order=max_order,)
            elif method == "hybrid":
                room = pra.ShoeBox(
                    p=room_dim,
                    fs=self.fs,
                    materials=pra.Material(e_absorption),
                    max_order=3,
                    ray_tracing=True,
                    air_absorption=True,)
        else:
            room = pra.ShoeBox(
            p=room_dim, fs=self.fs, **kwargs)
            if method == 'hybrid':
                room.set_ray_tracing(True)
                # room.set_air_absorption(True)
        
        for pos in src_pos:
            room.add_source(pos)
        room.add_microphone_array(mic_pos)
        room.compute_rir()

        self.c = room.c
        self.rir = room.rir


    def compute_srir_gpuRIR( 
        self, room_dim, src_pos, rt60, 
        mic_pos_center=None, 
        abs_weights = [0.9]*5+[0.5], 
        att_diff = 15.0, 
        att_max = 60.0,
        **kwargs):
        """Compute an SRIR from a given room parameters
            using gpuRIR

        Parameters
        ----------
        room_dim : (3,) array_like
            Room dimensions in meters
        src_pos : (num_src, 3) array_like
            Source positions in meters
        rt60 : float,
            Desired RT60 in seconds
        abs_weights : list, optional
            Absorption coefficient ratios of the walls,
            by default [0.9]*5+[0.5]
        att_diff : float, optional
            Desired attenuation (in dB), by default 15.0 dB
        att_max : float, optional
            Maximum attenuation (in dB), by default 60.0 dB
        mic_pos_center : (3,) array_like, optional
            Position of center of microphone array, 
            The default None, which means the center of room.
        kwargs : dict
            Additional arguments for gpuRIR.simulateRIR

        Returns
        -------
        array_like, shape (num_src, num_mic, length)
            Generated SRIR.
        """        
        
        import gpuRIR

        if mic_pos_center is None:
            mic_pos = self.mic_pos_cart + room_dim / 2
        else:
            mic_pos = self.mic_pos_cart + mic_pos_center
        room_dim = np.array(room_dim)
        src_pos = np.array(src_pos)

        beta = gpuRIR.beta_SabineEstimation(room_dim, rt60, abs_weights=abs_weights) # Reflection coefficients
        Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, rt60) # Time to start the diffuse reverberation model [s]
        Tmax = gpuRIR.att2t_SabineEstimator(att_max, rt60)	 # Time to stop the simulation [s]
        nb_img = gpuRIR.t2n( Tdiff, room_dim )	# Number of image sources in each dimension
        rir = gpuRIR.simulateRIR(
            room_dim, beta, src_pos, mic_pos, nb_img, 
            Tmax, self.fs, Tdiff=Tdiff, **kwargs)
        self.rir = np.transpose(rir, (1, 0, 2))
    

    def compute_srir_smir(
        self, room_dim, src_pos, rt60, SH_order=1,
        mic_pos_center=None, order=10,  K=1., 
        matlab_dir='./SMIR-Generator'):
        
        """Compute an SRIR from a given room parameters
            using SMIR-Generator.

        Parameters
        ----------
        room_dim : (3,) array_like
            Room dimensions in meters
        src_pos : (num_src, 3) array_like
            Source positions in meters
        rt60 : float,
            Desired RT60 in seconds
        order : int, optional
            Reflection order, by default 10
        mic_pos_center : (3,) array_like, optional
            Position of center of microphone array, 
            The default None, which means the center of room.
        K : float, optional
            Oversampling factor
        SH_order : int, optional
            Spherical harmonic order, by default 1
        matlab_dir : str, optional
            Path to SMIR-Generator, by default './SMIR-Generator',
            download URL: https://github.com/ehabets/SMIR-Generator
            
        Returns
        -------
        array_like, shape (num_src, num_mic, length)
            Generated SRIR.
        """        

        assert os.path.isdir(matlab_dir), \
            'SMIR-Generator not found, please download it from https://github.com/ehabets/SMIR-Generator'
        

        import matlab
        import matlab.engine

        if mic_pos_center is None:
            mic_pos_center = room_dim / 2
        mic = self.mic_pos_sph / 180 * np.pi
        mic[:, 1] = np.pi/2 - mic[:, 1]
        eng = matlab.engine.start_matlab()
        eng.cd(matlab_dir)
        self.rir = []
        for src in src_pos:
            rir = eng.smir_generator(
                self.c, float(self.fs), matlab.double(mic_pos_center.tolist()), matlab.double(src.tolist()), 
                matlab.double(room_dim.tolist()), float(rt60), 'rigid', self.radius, 
                matlab.double(mic.tolist()), float(SH_order), 
                self.fs/2, K, order, nargout=1     
            )
            self.rir.append(rir)
        self.rir = np.transpose(self.rir, (1, 0, 2))
        eng.quit()



    def simulate(self, src_pos_mic, src_signals, n_points=2048, **kwargs):
        """Simulates the microphone signal at every microphone in the array
        
        """

        assert len(src_pos_mic) == len(src_signals), \
            "Number of source position and signals must be equal."
        assert self.rir is not None, "Room impulse response is not computed."

        num_src = len(src_pos_mic)
        num_mic = len(self.rir)

        max_len_rir = np.array(
            [len(self.rir[i][j]) for i, j in product(range(num_mic), range(num_src))]
        ).max()
        f = lambda i: len(src_signals[i])
        max_sig_len = np.array([f(i) for i in range(num_src)]).max()
        num_points = int(max_len_rir) + int(max_sig_len) - 1
        if num_points % 2 == 1:
            num_points += 1
        # the array that will receive all the signals
        premix_signals = np.zeros((num_src, num_mic, num_points))
        # compute the signal at every microphone in the array
        for m in np.arange(num_mic):
            for s in np.arange(num_src):
                sig = src_signals[s]
                if sig is None:
                    continue
                h = self.rir[m][s]
                premix_signals[s, m, :len(sig) + len(h) - 1] += \
                    scysignal.fftconvolve(h, sig)

        return np.sum(premix_signals, axis=0)
            


        

