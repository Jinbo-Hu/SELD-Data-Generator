# Spatial Room Impulse Response (SRIR) Generater

This repository is for generating spherical microphone array room impulse response in sound event localization and detection (SELD) task.

## Quick Start
The directory of sound event datasets should look like:

<pre>
./sound_event_datasets
├────bell
│        ├── Bicycle_bell
│        │   └── 29623.wav
│        ├── Chime
│        └── Doorbell
└────Clapping
        └─── Clapping
                ├── 2080.wav
                └── 2081.wav
</pre>

Then run: 
``` shell
python make_dataset.py 1
```