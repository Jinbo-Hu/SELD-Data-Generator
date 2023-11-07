# Reference: https://github.com/danielkrause/DCASE2022-data-generator
import functools
import os
from pathlib import Path

import librosa
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import utils
from srir.srir import GenerateSRIR as SRIR


class DataSynthesizer(object):
    def __init__(self, db_config, params):
        self._db_config = db_config
        self.params = params
        self.max_samples_per_cls = params['nb_events_per_classes']
        self.max_polyphony = {
            'target_classes': params['max_polyphony_target'],
            'interf_classes': params['max_polyphony_interf'],
        }
        self._metadata_path = params['mixturepath'] + '/' + 'metadata'
        self._mixture_path = {
            'mic': params['mixturepath'] + '/' + 'mic',
        }
        self._classnames = db_config._classes
        self._active_classes = {
            'target_classes': np.sort(params['target_classes']),
            'interf_classes': np.sort(params['interf_classes'])
        }
        self._nb_active_classes = {
            'target_classes': len(self._active_classes['target_classes']),
            'interf_classes': len(self._active_classes['interf_classes'])
        } 

        # self._class_mobility = db_config._class_mobility
        self._mixture_setup = {}
        self._mixture_setup['classnames'] = []
        for cl in self._classnames:
            self._mixture_setup['classnames'].append(cl)
        self._apply_gains = True
        self._class_gains = db_config._sample_list['energy_quartile']
        self._mixture_setup['fs_mix'] = 24000 #fs of RIRs
        self._mixture_setup['mixture_duration'] = params['mixture_duration']
        self._mixture_setup['mixture_points'] = int(self._mixture_setup['fs_mix'] * params['mixture_duration'])
        self._nb_mixtures = params['nb_mixtures']
        self._mixture_setup['total_duration'] = self._nb_mixtures * self._mixture_setup['mixture_duration']
        self._mixture_setup['snr_set'] = np.arange(6.,31.)
        self._mixture_setup['time_idx_100ms'] = np.arange(0.,self._mixture_setup['mixture_duration'],0.1)
        self._mixture_setup['start_delay'] = np.arange(0., params['start_delay'], 0.1)
        #### SRIR setup #####
        self._mixture_setup['room_size_range'] = np.asarray(params['room_size_range'])
        self._mixture_setup['temperature_range'] = np.asarray(params['temperature_range'])
        self._mixture_setup['humidity_range'] = np.asarray(params['humidity_range'])
        self._mixture_setup['RT60_range'] = np.asarray(params['RT60_range'])
        self._mixture_setup['mic_pos_range_percentage'] = np.asarray(params['mic_pos_range_percentage'])
        self._mixture_setup['src_pos_from_walls'] = params['src_pos_from_walls']

        self._nb_frames = len(self._mixture_setup['time_idx_100ms'])
        self._rnd_generator = np.random.default_rng(seed=params['seed'])
        self._nb_classes = len(self._classnames)
        self._nb_snrs = len(self._mixture_setup['snr_set'])
        self._nb_dealys = len(self._mixture_setup['start_delay'])

        self._mixtures = {
            'target_classes': [],
            'interf_classes': [],
        }
        self._metadata = {
            'target_classes': [],
            'interf_classes': [],
        }
        self._srir_setup = {
            'target_classes': [],
            'interf_classes': [],
        }
        self._noise_path = {
            '01_bomb_center': 39840000,
            '02_gym': 24960000,
            '03_pb132_paatalo_classroom2': 44640000,
            '04_pc226_paatalo_office': 29592000,
            '05_sa203_sahkotalo_lecturehall': 30000000,
            '06_sc203_sahkotalo_classroom2': 30984000,
            '08_se203_sahkotalo_classroom': 44784000,
            '09_tb103_tietotalo_lecturehall': 59280000,
            '10_tc352_tietotalo_meetingroom': 33480000,
        }
    
    def create_mixtures(self, scenes='target_classes'):
        """ Create mixtures for the target and interf class index.
        """
        
        foldlist = {}

        print('\nGenerating mixtures...\n')

        idx_active1 = np.array([])
        idx_active2 = np.array([])
        path_dict = dict()
        for na in range(self._nb_active_classes[scenes]):
            idx_active1 = np.append(idx_active1, \
                np.nonzero(self._db_config._sample_list['class'] == self._active_classes[scenes][na]))
        # pick out nb_samples_per_cls samples each subclasses
        for idx, path in enumerate(self._db_config._sample_list['audiofile']):
            fn = str(path).split('/')[-4:]
            if fn[0] not in path_dict.keys():
                path_dict[fn[0]] = dict()
            if fn[2] not in path_dict[fn[0]].keys():
                path_dict[fn[0]][fn[2]] = np.array([])
            path_dict[fn[0]][fn[2]] = np.append(path_dict[fn[0]][fn[2]], idx)
        for cls in path_dict.keys():
            num_subcls = len(path_dict[cls].keys())
            max_sample_per_subcls = int(np.floor(self.max_samples_per_cls[self._classnames.index(cls)] / num_subcls))
            for subcls in path_dict[cls].keys():
                sampleperm = self._rnd_generator.permutation(len(path_dict[cls][subcls]))
                path_dict[cls][subcls] = path_dict[cls][subcls][sampleperm]
                num_tile = int(np.floor(max_sample_per_subcls / len(path_dict[cls][subcls])))
                if num_tile > 1:
                    subcls_path_idx = np.tile(path_dict[cls][subcls], num_tile)
                else:
                    subcls_path_idx = path_dict[cls][subcls]
                idx_active2 = np.append(idx_active2, subcls_path_idx[:max_sample_per_subcls])
        idx_active1 = idx_active1.astype('int')
        idx_active2 = idx_active2.astype('int')
        # intersection set
        idx_active = np.intersect1d(idx_active1, idx_active2)

        foldlist['class'] = self._db_config._sample_list['class'][idx_active]
        foldlist['audiofile'] = self._db_config._sample_list['audiofile'][idx_active]
        foldlist['duration'] = self._db_config._sample_list['duration'][idx_active]
        foldlist['onoffset'] = self._db_config._sample_list['onoffset'][idx_active]
        nb_samples = len(foldlist['duration'])

        sampleperm = self._rnd_generator.permutation(nb_samples)
        foldlist['class'] = foldlist['class'][sampleperm]
        foldlist['audiofile'] = foldlist['audiofile'][sampleperm]
        foldlist['duration'] = foldlist['duration'][sampleperm]
        foldlist['onoffset'] = foldlist['onoffset'][sampleperm]
        
        iterator = tqdm(range(self._nb_mixtures), total=self._nb_mixtures, unit='mixtures')
        sample_idx = 0
        for _ in iterator:
            mixture = {}
            mixture['class'] = []
            mixture['audiofile'] = []
            mixture['duration'] = []
            mixture['onoffset'] = []
            mixture['start_time'] = []

            for _ in range(self.max_polyphony[scenes]):
                start_time = self._rnd_generator.choice(self._mixture_setup['start_delay'])
                trim_length = foldlist['duration'][sample_idx] + start_time - \
                    self._mixture_setup['mixture_duration']
                onset = foldlist['onoffset'][sample_idx][0]
                offset = foldlist['onoffset'][sample_idx][1] - trim_length \
                    if trim_length > 0 else foldlist['onoffset'][sample_idx][1]

                mixture['class'].append(foldlist['class'][sample_idx])
                mixture['audiofile'].append(foldlist['audiofile'][sample_idx])
                mixture['duration'].append(offset - onset)
                mixture['onoffset'].append([onset, offset])
                mixture['start_time'].append(start_time)

                sample_idx += 1
                if sample_idx == nb_samples:
                    sample_idx = 0
            
            self._mixtures[scenes].append(mixture)

        iterator.close()

    def create_metadata(self, add_interf=True):
        """ Create metadata for the mixture.
        """
        # NOTE: it only supports static sources.

        print('\n Preparing metadata...\n')

        room_size = self._rnd_generator.uniform(
            low=self._mixture_setup['room_size_range'][:, 0], 
            high=self._mixture_setup['room_size_range'][:, 1], 
            size=(self._nb_mixtures, 3)
        )
        mic_pos_percentage = self._rnd_generator.uniform(
            low=self._mixture_setup['mic_pos_range_percentage'][0],
            high=self._mixture_setup['mic_pos_range_percentage'][1], 
            size=self._nb_mixtures
        )
        rt60 = self._rnd_generator.uniform(
            low=self._mixture_setup['RT60_range'][0], 
            high=self._mixture_setup['RT60_range'][1], 
            size=self._nb_mixtures
        )

        for nmix in range(self._nb_mixtures):
            nth_metadata = {
                'classid': [None] * self._nb_frames, 
                'trackid': [None] * self._nb_frames, 
                'eventtimetracks': [None] * self._nb_frames, 
                'eventdoatimetracks': [None] * self._nb_frames
            }
            nmix_setup = {
                'room_size': None, 
                'mic_pos_center': None, 
                'src_pos': [], 
                'rt60':None
            }

            nmix_room_size = room_size[nmix]
            nmix_mic_pos_center = mic_pos_percentage[nmix] * nmix_room_size
            nmix_rt60 = rt60[nmix]

            while True:
                try:
                    pra.inverse_sabine(nmix_rt60, nmix_room_size)
                except ValueError:
                    nmix_room_size = self._rnd_generator.uniform(
                        low=self._mixture_setup['room_size_range'][:, 0], 
                        high=self._mixture_setup['room_size_range'][:, 1],
                    )
                    print('ValueError: rt60[{}] = {} for room_size {}'\
                        .format(nmix, nmix_rt60, nmix_room_size))
                else:
                    break

            nmix_setup['room_size'] = nmix_room_size
            nmix_setup['mic_pos_center'] = nmix_mic_pos_center
            nmix_setup['rt60'] = nmix_rt60

            for nlayer in range(self.max_polyphony['target_classes']):
                nlayer_src_pos = self._rnd_generator.uniform(
                    low=self._mixture_setup['src_pos_from_walls'],
                    high=nmix_room_size-self._mixture_setup['src_pos_from_walls']
                )
                x, y, z = nlayer_src_pos - nmix_mic_pos_center
                azi, ele, _ = np.squeeze(utils.cart2sph(x, y, z))
                nmix_setup['src_pos'].append(nlayer_src_pos)

                start_idx = np.ceil(
                    (self._mixtures['target_classes'][nmix]['start_time'][nlayer] + \
                    self._mixtures['target_classes'][nmix]['onoffset'][nlayer][0]) / 0.1)
                end_idx = np.ceil(
                    (self._mixtures['target_classes'][nmix]['start_time'][nlayer] + \
                    self._mixtures['target_classes'][nmix]['onoffset'][nlayer][1]) / 0.1)
                end_idx = min(end_idx, self._nb_frames)

                for frame_idx in range(int(start_idx), int(end_idx)):
                    if nth_metadata['classid'][frame_idx] is None:
                        nth_metadata['classid'][frame_idx] = \
                            [self._mixtures['target_classes'][nmix]['class'][nlayer]]
                        nth_metadata['trackid'][frame_idx] = [nlayer]
                        nth_metadata['eventtimetracks'][frame_idx] = \
                            [self._mixtures['target_classes'][nmix]['start_time'][nlayer]]
                        nth_metadata['eventdoatimetracks'][frame_idx]= [[azi, ele]]
                    else:
                        nth_metadata['classid'][frame_idx].append(
                            self._mixtures['target_classes'][nmix]['class'][nlayer])
                        nth_metadata['trackid'][frame_idx].append(nlayer)
                        nth_metadata['eventtimetracks'][frame_idx].append(
                            self._mixtures['target_classes'][nmix]['start_time'][nlayer])
                        nth_metadata['eventdoatimetracks'][frame_idx].append([azi, ele])        
    
            self._metadata['target_classes'].append(nth_metadata)       
            self._srir_setup['target_classes'].append(nmix_setup)

            # Add interference source
            if add_interf:
                nmix_setup_interf = {'src_pos': []}
                for nlayer in range(self.max_polyphony['interf_classes']):
                    nlayer_src_pos = self._rnd_generator.uniform(
                        low=self._mixture_setup['src_pos_from_walls'],
                        high=nmix_room_size-self._mixture_setup['src_pos_from_walls']
                    )
                    nmix_setup_interf['src_pos'].append(nlayer_src_pos)
                self._srir_setup['interf_classes'].append(nmix_setup_interf)


    def write_metadata(self, scenes='target_classes'):
        r""" Write metadata for the mixture.
        """

        if scenes == 'interf_classes':
            return

        if not os.path.isdir(self._metadata_path):
            Path(self._metadata_path).mkdir(exist_ok=True, parents=True)
        
        print('\n Writing metadata...\n')

        iterator = tqdm(range(self._nb_mixtures), total=self._nb_mixtures, unit='mixtures')
        for nmix in iterator:
            print('Writing metadata for mixture {}/{}...'.format(nmix, self._nb_mixtures))
            mixture = self._metadata[scenes][nmix]
            mixture_name = 'fold0_room0_mix{}.csv'.format(nmix)
            file_id = open(os.path.join(self._metadata_path, mixture_name), 'w')
            for frame_idx in range(self._nb_frames):
                if mixture['classid'][frame_idx] is None:
                    continue
                num_events = len(mixture['classid'][frame_idx])
                for event_idx in range(num_events):
                    classid = mixture['classid'][frame_idx][event_idx]
                    classid = self.params[scenes].index(classid)
                    azi, ele = mixture['eventdoatimetracks'][frame_idx][event_idx]
                    file_id.write('{},{},{},{},{}\n'.format(
                        frame_idx, classid, event_idx, int(azi), int(ele)))
            file_id.close()
        iterator.close()
    
    def synthesize_mixtures(self, add_interf=True, audio_format='both', add_noise=True):
        r""" Synthesize mixtures.
        """
        assert audio_format in ['both', 'foa', 'mic'], \
            'audio_format must be either "both", "foa" or "mic".'
        
        for _subdir in self._mixture_path.keys():
            if not os.path.isdir(self._mixture_path[_subdir]):
                Path(self._mixture_path[_subdir]).mkdir(exist_ok=True, parents=True)

        process_map(
            functools.partial(
                self.generate_mixture,
                self._mixtures,
                self._srir_setup,
                add_interf,
                add_noise,
                audio_format,
            ),
                range(self._nb_mixtures),
                max_workers=self.params['max_workers'],
                chunksize=self.params['chunksize'],
        )   

    def generate_mixture(self, mixtures, srir_setups, add_interf, add_noise, audio_format, nmix):
        """ Write mixture to disk.

        """
        mixture_name = 'fold0_room0_mix{}.wav'.format(nmix)

        mixture = mixtures['target_classes'][nmix]
        srir_setup = srir_setups['target_classes'][nmix]

        room_size = srir_setup['room_size']
        target_audio = mixture['audiofile']
        src_pos = srir_setup['src_pos']
        mic_pos_center = srir_setup['mic_pos_center']
        rt60 = srir_setup['rt60']

        srir_generator = SRIR(
            fs=self._mixture_setup['fs_mix'],
            mic_pos=self.params['mic_pos'],
            radius=self.params['radius'],
        )
        
        src_sig = []
        for event_id, file in enumerate(target_audio):
            onset, offset = mixture['onoffset'][event_id]
            duration = mixture['duration'][event_id]
            start_time = mixture['start_time'][event_id]
            audio, fs = librosa.load(
                path=file, 
                sr=self._mixture_setup['fs_mix'], 
                offset=onset, 
                duration=duration
            )
            audio = utils.segment_mixtures(
                signal=audio,
                fs=fs, 
                start=onset+start_time, 
                end=offset+start_time, 
                clip_length=self._mixture_setup['mixture_duration']
            )
            if self._apply_gains:
                audio = utils.apply_event_gains(
                    audio, onset, offset, self._class_gains, mixture['class'][event_id])
            src_sig.append(audio)
        
        if add_interf:
            mixture_interf = mixtures['interf_classes'][nmix]
            srir_setup_interf = srir_setups['interf_classes'][nmix]
            interf_audio = mixture_interf['audiofile']
            src_pos.extend(srir_setup_interf['src_pos'])

            for event_id, file in enumerate(interf_audio):
                onset, offset = mixture_interf['onoffset'][event_id]
                duration = mixture_interf['duration'][event_id]
                start_time = mixture_interf['start_time'][event_id]
                audio, fs = librosa.load(
                    path=file, 
                    sr=self._mixture_setup['fs_mix'], 
                    offset=onset, 
                    duration=duration
                )
                audio = utils.segment_mixtures(
                    signal=audio, 
                    fs=fs, 
                    start=onset+start_time, 
                    end=offset+start_time, 
                    clip_length=self._mixture_setup['mixture_duration']
                )
                if self._apply_gains:
                    audio = utils.apply_event_gains(
                        audio, onset, offset, self._class_gains, mixture_interf['class'][event_id])
                src_sig.append(audio)

        if self.params['tools'] in ['pyroomacoustics', 'gpuRIR', 'smir']:
            srir_generator.compute_srir(
                rt60=rt60, 
                room_dim=room_size, 
                src_pos=src_pos,
                method=self.params['method'],
                mic_pos_center=mic_pos_center,
            )
        else:
            raise ValueError('Unknown tools for SRIR generation.')

        audio_mic = srir_generator.simulate(src_pos_mic=src_pos-mic_pos_center, src_signals=src_sig)
        audio_mic = audio_mic[:, :self._mixture_setup['mixture_points']]
        if add_noise:
            noise_path = self._rnd_generator.choice(list(self._noise_path.keys()))
            path = '{}/{}/ambience_tetra_24k_edited.wav'.format(self.params['noisepath'], noise_path)
            start_idx = self._rnd_generator.choice(range(0, self._noise_path[noise_path] - self._mixture_setup['mixture_points']))
            # ambience, _ = sf.read(path, start=start_idx, frames=self._mixture_setup['mixture_points'])
            ambience, _ = librosa.load(
                path=path, 
                sr=self._mixture_setup['fs_mix'],
                offset=start_idx / self._mixture_setup['fs_mix'], 
                duration=self._mixture_setup['mixture_duration'],
                mono=False,
            )
            audio_energy = np.sum(np.mean(audio_mic, axis=0)**2)
            ambience_energy = np.sum(np.mean(ambience, axis=0)**2)
            snr = self._rnd_generator.choice(self._mixture_setup['snr_set'])
            ambi_norm = np.sqrt(audio_energy * (10.**(-snr/10.)) / ambience_energy)
            audio_mic += ambi_norm * ambience
            
        clip_path_mic = os.path.join(self._mixture_path['mic'], mixture_name)
        if audio_format in ['mic', 'both']:
            sf.write(file=clip_path_mic, data=audio_mic.T, samplerate=self._mixture_setup['fs_mix'])
        if audio_format in ['foa', 'both']:
            # TODO: add FOA format
            pass
        tqdm.write(mixture_name)
