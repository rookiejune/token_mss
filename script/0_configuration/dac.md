# descript-audio-codec[https://github.com/descriptinc/descript-audio-codec/tree/main]

## audiotools[https://github.com/descriptinc/audiotools]

## Install

`pip install descript-audio-codec`

To download pretrained model:

`python3 -m dac download # downloads the default 44kHz variant
python3 -m dac download --model_type 44khz # downloads the 44kHz variant
python3 -m dac download --model_type 24khz # downloads the 24kHz variant
python3 -m dac download --model_type 16khz # downloads the 16kHz variant`

## Encode

`python3 -m dac encode /path/to/input --output /path/to/output/codes`


## Decode

`python3 -m dac decode /path/to/output/codes --output /path/to/reconstructed_input`