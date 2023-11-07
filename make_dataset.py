# Reference: https://github.com/danielkrause/DCASE2022-data-generator
import os
import pickle
import sys

from data_generator.data_synthesis import DataSynthesizer
from data_generator.db_config import DBConfig
from get_parameters import get_params


def main(arg):

    taskid = int(arg[1])
    params = get_params(arg)
    print('\n TASK-ID: {}\n'.format(taskid))

    db_config_path = './db_config_{}.obj'.format(params['db_name'])
    ### Create database config based on params (e.g. filelist name etc.)
    if not os.path.isfile(db_config_path):
        db_config = DBConfig(params)
        # WRITE DB-config
        with open(db_config_path, 'wb') as f:
            pickle.dump(db_config, f)
        print('################')
        print(db_config_path, 'has been saved!')
        # sys.exit()
    else:    
        # LOAD DB-config which is already done
        with open(db_config_path, 'rb') as f:
            db_config = pickle.load(f)
        print('################')
        print(db_config_path, 'has been loaded!')
    
    data_synth = DataSynthesizer(db_config, params)
    
    data_synth.create_mixtures(scenes='target_classes')
    data_synth.create_mixtures(scenes='interf_classes')
    data_synth.create_metadata(add_interf=params['add_interf'])

    data_synth.write_metadata(scenes='target_classes')
    data_synth.synthesize_mixtures(
        add_interf=params['add_interf'], 
        audio_format=params['audio_format'],
        add_noise=params['add_noise'])


if __name__ == '__main__':
    main(sys.argv)
