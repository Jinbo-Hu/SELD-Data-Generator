# Reference: https://github.com/danielkrause/DCASE2022-data-generator
import numpy as np
import librosa
from pathlib import Path
from multiprocessing import Manager
from tqdm.contrib.concurrent import process_map
import functools

class DBConfig(object):
    def __init__(self, params):
        self._mixturepath = params['mixturepath']
        self._db_path = params['db_path']
        self._db_name = params['db_name']
        self._rnd_generator = np.random.default_rng()
        self._classes = ['femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter', 'domesticSounds', 
                        'footsteps','doorCupboard', 'music', 'musicInstrument', 'waterTap', 'bell', 'knock', 
                         'computerKeyboard', 'shufflingCards', 'dishesPotsAndPans']
        self._nb_classes = len(self._classes)
        self._class_mobility = [2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1]
        self._apply_class_gains = True
        self._class_gains = None
        self._sample_list = self._load_db_fileinfo()     


    def load_file(self, file_list, sample_list, start, ncl, ns):
        audio_path = file_list[ncl][ns]
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio)/float(sr)
        if len(audio) > 0.1 * sr:
            sample_list['class'][start+ns] = ncl
            sample_list['audiofile'][start+ns] = str(file_list[ncl][ns])
            sample_list['duration'][start+ns] = duration
            sample_list['onoffset'][start+ns] = [0., duration]
            pow_per_sec = np.sum(audio**2) / duration
            sample_list['energy_per_sec'][start+ns] = pow_per_sec
            
    def _load_db_fileinfo(self):
        file_list = self._make_selected_filelist()
        manager = Manager()

        print('Preparing sample list...')
        sample_list = manager.dict()
        sample_list.update({'class': manager.list(), 'audiofile': manager.list(), 'duration': manager.list(),
                            'onoffset': manager.list(), 'nSamplesPerClass': np.array([]), 
                            'meanStdDurationPerClass': np.array([]), 'minMaxDurationPerClass': np.array([]),
                            'energy_per_sec': manager.list(), 'energy_quartile': np.array([])})
        start_idx = 0
        for ncl in range(self._nb_classes):
            counter = 0
            nb_samples_per_class = len(file_list[ncl])
            sample_list['class'].extend([None]*nb_samples_per_class)
            sample_list['audiofile'].extend([None]*nb_samples_per_class)
            sample_list['duration'].extend([None]*nb_samples_per_class)
            sample_list['onoffset'].extend([None]*nb_samples_per_class)
            sample_list['energy_per_sec'].extend([None]*nb_samples_per_class)
            process_map(
                functools.partial(
                    self.load_file,
                    file_list, sample_list, start_idx, ncl),
                    range(nb_samples_per_class),
                    max_workers=16,
                    chunksize=16,
                    desc='class'+str(ncl),
                )
            while None in sample_list['class']:
                counter += 1
                sample_list['class'].remove(None)
                sample_list['audiofile'].remove(None)
                sample_list['duration'].remove(None)
                sample_list['onoffset'].remove(None)
                sample_list['energy_per_sec'].remove(None)
            start_idx += nb_samples_per_class - counter 

        sample_list['class'] = np.array(sample_list['class'])
        sample_list['audiofile'] = np.array(sample_list['audiofile'])
        sample_list['duration'] = np.array(sample_list['duration'])
        sample_list['onoffset'] = np.squeeze(np.array(sample_list['onoffset'],dtype=object))
        sample_list['energy_per_sec'] = np.array(sample_list['energy_per_sec'])
        
        for n_class in range(self._nb_classes):
            class_idx = (sample_list['class'] == n_class)
            sample_list['nSamplesPerClass'] = np.append(sample_list['nSamplesPerClass'], np.sum(class_idx))
            energy_per_sec = np.array(sample_list['energy_per_sec'])
            if n_class == 0:
                sample_list['meanStdDurationPerClass'] = \
                    np.array([[np.mean(sample_list['duration'][class_idx]), 
                               np.std(sample_list['duration'][class_idx])]])
                sample_list['minMaxDurationPerClass'] =  \
                    np.array([[np.min(sample_list['duration'][class_idx]), 
                               np.max(sample_list['duration'][class_idx])]])
                sample_list['energy_quartile'] = \
                    np.array([np.min(energy_per_sec), np.quantile(energy_per_sec, 0.25),
                              np.median(energy_per_sec), np.quantile(energy_per_sec, 0.75), 
                              np.max(energy_per_sec)])
            else:
                sample_list['meanStdDurationPerClass'] = \
                    np.vstack((sample_list['meanStdDurationPerClass'], 
                               np.array([np.mean(sample_list['duration'][class_idx]), 
                                         np.std(sample_list['duration'][class_idx])])))
                sample_list['minMaxDurationPerClass'] = \
                    np.vstack((sample_list['minMaxDurationPerClass'],
                               np.array([np.min(sample_list['duration'][class_idx]), 
                                         np.max(sample_list['duration'][class_idx])])))
                sample_list['energy_quartile'] = \
                    np.vstack((sample_list['energy_quartile'], 
                               np.array([np.min(energy_per_sec), np.quantile(energy_per_sec, 0.25), 
                                         np.median(energy_per_sec), np.quantile(energy_per_sec, 0.75), 
                                         np.max(energy_per_sec)])))
                
        return dict(sample_list)
    
    def _make_selected_filelist(self):
        file_list = []
        class_list = self._classes #list(self._classes.keys())
        
        for ntc in range(self._nb_classes):
            classpath = Path(self._db_path + '/' + class_list[ntc])
            filelist = [file for file in classpath.glob('**/*.wav')]
            file_list.append(filelist)
        return file_list
