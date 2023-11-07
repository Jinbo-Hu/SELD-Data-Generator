# Reference: https://github.com/danielkrause/DCASE2022-data-generator
import os
from pathlib import Path
import warnings
import numpy as np


# Parameters used in the data generation process.
def get_params(arg=0):
    task_id = int(arg[1])
    params = dict(
        # path containing background noise recordings
        noisepath = '???', 
        mixture_duration = 5., # seconds
        start_delay = 3., # seconds
        nb_mixtures = 100000, # number of mixtures per fold
        audio_format = 'both', # 'foa' , 'mic' or 'both'
        mixturepath = '???', # root path to the mixture files
        db_name = '???', # run data_generator/db_config.py .obj file
        db_path = '???', # path to the dataset
        nb_events_per_classes = [1000] * 16,
        range_events_per_classes = [[0., 1.]] * 16, # range of all events for each class
        max_polyphony_target = 3,
        max_polyphony_interf = 1,
        target_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        interf_classes = [13, 14, 15],
        seed = 2022, # fix the seed for reproducibility
        chunksize = 1024,
        max_workers = 128,
        ################ SRIR Parameters ################
        #### mic array parameters #####
        radius = 0.042, # radius of the spherical array
        mic_pos = [[45,35],[-45,-35],[135,-35],[-135,35]],
        #### Room Parameters #####
        # [[lx_min, lx_max],[ly_min, ly_max],[lz_min, lz_max]]
        room_size_range = [[5., 25.], [5., 25.], [3., 10.]],
        # [value_min, value_max]
        temperature_range = [15, 35], # degree Celsius
        humidity_range = [0, 100],
        RT60_range = [0.2, 2], # in seconds
        mic_pos_range_percentage = [0.4, 0.6], # percentage of the room size
        src_pos_from_walls = 1,
        method = 'hybrid', # 'hybrid' or 'ism'
        tools = 'pyroomacoustics', # 'gpuRIR' or 'pyroomacoustics' or 'smir'
        add_noise = False,
        add_interf = False,
    )

    if task_id == 0:
        ################################################################################
        #### Default For DCASE 2022-2023 Task 3
        ################################################################################
        params['out_putdir'] = 'simulate'
        params['mixturepath'] += params['out_putdir']
        params['audio_format'] = 'foa'
    params['mic_pos'] = np.array(params['mic_pos'])

    if params['tools'] != 'pyroomacoustics' and params['method'] == 'hybrid':
        warnings.warn('Hybrid method only support pyroomacoustics, change to ISM!')
        params['method'] = 'ism'


    Path(params['mixturepath']).mkdir(parents=True, exist_ok=True)
    param_path = os.path.join(params['mixturepath'], 'params.txt')
    f = open(param_path, 'w')
    for key, value in params.items():
        word = "\t{}: {}\n".format(key, value)
        f.writelines(word)
        print(word)
    f.close()

    return params

if __name__ == '__main__':
    get_params()