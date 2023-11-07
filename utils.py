# Reference: https://github.com/danielkrause/DCASE2022-data-generator
import numpy as np 
import pyroomacoustics as pra


def sph2cart(azimuth, elevation, r, type='degree'):
    r"""
    Convert spherical to cartesian coordinates
    """
    assert type in ['degree', 'radian'], "Type must be 'degree' or 'radian'"
    if type == 'degree':
        azimuth = azimuth / 180.0 * np.pi
        elevation = elevation / 180.0 * np.pi

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)

    return np.c_[x, y, z]


def cart2sph(x, y, z, type='degree'):
    r"""
    Convert cartesian to spherical coordinates    
    """
    assert type in ['degree', 'radian'], "Type must be 'degree' or 'radian'"

    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)

    if type == 'degree':
        azimuth = azimuth / np.pi * 180.0
        elevation = elevation / np.pi * 180.0

    return np.c_[azimuth, elevation, r]


def doa_estimate(
    mic_pos, receiver, fs=24000, nfft=256, num_src=2, freq_bins=np.arange(5,125), dim=3, c=343.0
    ):
    r"""
    Estimate DOA from signals
    """

    ################################
    # Compute the STFT frames needed
    X = np.array(
        [
            pra.transform.stft.analysis(signal, nfft, nfft // 2).T
            for signal in receiver
        ]
    )

    ##############################################
    # Now we can test all the algorithms available
    # algo_names = sorted(pra.doa.algorithms.keys())
    algo_names = ["SRP", "NormMUSIC"]
    for algo_name in algo_names:
        # Construct the new DOA object
        doa = pra.doa.algorithms[algo_name](mic_pos, fs, nfft, dim=dim, c=c, num_src=num_src)

        # this call here perform localization on the frames in X
        doa.locate_sources(X, freq_bins=freq_bins)

        # doa.azimuth_recon contains the reconstructed location of the source
        print(algo_name)
        print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
        if dim == 3:
            print("  Recovered elevation:", - doa.colatitude_recon / np.pi * 180.0 + 90.0, "degrees")


def asarray_1d(a, **kwargs):
    r"""Squeeze the input and check if the result is one-dimensional.

    Returns *a* converted to a `numpy.ndarray` and stripped of
    all singleton dimensions.  Scalars are "upgraded" to 1D arrays.
    The result must have exactly one dimension.
    If not, an error is raised.
    """
    result = np.squeeze(np.asarray(a, **kwargs))
    if result.ndim == 0:
        result = result.reshape((1,))
    elif result.ndim > 1:
        raise ValueError("array must be one-dimensional")
    return result


def segment_mixtures(signal, fs, start, end, clip_length=5):
    r"""
    If the duration of the signal is less than 5 seconds, pad the signal with zeros at the beginning and
    end. Otherwise, return the first 5 seconds of the signal
    """
    
    duration = np.shape(signal)[0] / fs
    if duration < clip_length:
        pad_width_before = int(start * fs)
        pad_width_after = max(0, int(fs*(clip_length-end)))
        pad_width = ((pad_width_before, pad_width_after),)
        return np.pad(signal, pad_width)
    else:
        return signal[:int(clip_length*fs)]


def sample_from_quartiles(K, stats):
    r"""
    Uniformly sample K points from the quartiles of the distribution.
    
    :param K: number of sampling points
    :param stats: a list of the 5-number summary of the data. 
                  stat = [min, quart1, median, quart3, max].
    :return: a list of samples from the given data.
    """

    minn = stats[0]
    maxx = stats[4]
    quart1 = stats[1]
    mediann = stats[2]
    quart3 = stats[3]
    samples = minn + (quart1 - minn)*np.random.rand(K, 1)
    samples = np.append(samples,quart1)
    samples = np.append(samples, quart1 + (mediann-quart1)*np.random.rand(K,1))
    samples = np.append(samples,mediann)
    samples = np.append(samples, mediann + (quart3-mediann)*np.random.rand(K,1))
    samples = np.append(samples, quart3)
    samples = np.append(samples, quart3 + (maxx-quart3)*np.random.rand(K,1))
    
    return samples


def apply_event_gains(signal, onset, offset, class_gains, class_idx):
    r"""Apply event gains of class_idx to the signal.
    """

    K=1000
    rand_energies_per_spec = sample_from_quartiles(K, class_gains[class_idx])
    # intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(3*(K+1))]
    intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(2*(K+1))]
    rand_energy_per_spec = intr_quart_energies_per_sec[np.random.randint(len(intr_quart_energies_per_sec))]
    sample_active_time = offset - onset

    target_energy = rand_energy_per_spec*sample_active_time
    event_energy = np.sum(signal**2)
    norm_gain = np.sqrt(target_energy/(event_energy+1e-10))

    return norm_gain * signal

