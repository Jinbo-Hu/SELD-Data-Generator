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

```shell
python make_dataset.py 1
```

## Others

FOA Converter has not yet fully completed. We will also release codes of this part as soon as possible.

## Reference

[1] Jinbo Hu, Yin Cao, Ming Wu, Qiuqiang Kong, Feiran Yang, Mark D. Plumbley, and Jun Yang, "Selective-Memory Meta-Learning with Environment Representations for Sound Event Localization and Detection" *arXiv* *preprint* *arXiv*:2312.16422, 2023. [URL](https://arxiv.org/abs/2312.16422)

## External Links

1. [https://github.com/danielkrause/DCASE2022-data-generator](https://github.com/danielkrause/DCASE2022-data-generator)
2. [https://github.com/ehabets/SMIR-Generator](https://github.com/ehabets/SMIR-Generator)
