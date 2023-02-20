# MP3 Project 2023

## Installation Guide

To install this package first clone this repository:

```git clone https://github.com/marios1861/mp3.git```

cd into the created directory

```cd mp3```

then create a virtual environment:

```python -m venv venv```

activate the virtual environment:

```. ./venv/bin/activate```

install the project and its requirements:

```pip install -r requirements.txt```

You are now ready to run the scripts described below!

## Test Scripts

There are 3 test cases included in this package:

>```mp3convert```

>```mp3psycho```

>```mp3quantum```

### ```mp3convert```

Showcases the finite impulse response filters used in mp3, by filtering, downsampling and then upsampling and filtering the samples from an example wav file. Also includes the total implementation of the mp3 standard. Get help by typing `mp3convert -h`.

### ```mp3psycho```

Includes tests for the mp3 psychoacoustic model. The goal is to test the programmatic correctness of the contituent functions, not the actual quality of the global thresholds produced.

### ```mp3quantum```

Includes tests for the psychoacoustic model-aware quantization, dequantization, run-length encoding and huffman encoding. Assert statements ensure correctness.
