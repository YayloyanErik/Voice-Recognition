# Voice-Recognition
Welcome to the Voice Recognition repository!

## Table of Contents

- [Overview](#overview)
- [About](#about)
- [Installation](#installation)
- [Getting Started](#getting-started)


## Overview
This simple catboost model is trained on audios of 3 speaking people and will recognise the speaker.

## About
All functions to create a model are located in `AudioRecognizing.py` and they runs in `model_creation.ipynb`
1. Using `os` library it creates folders for script

2. It converts audios from stereo to mono

3. Then it cleans data's parts that are lower than mean/3 (mean is data's arithmetic mean)

4. Using `Split()`(from `AudioRecognizing.py`) function it split audio to 100ms(recomended) segments and calculates min segments

5. Using FFT it gets magnitudes of segments and create csv files for every person

6. Finally it creates CatBoost model from given 3 csv files, model iterations, path where model will be saved, verbose(optional)

## Installation

To install this project, clone the repository and run following commands:

```bash
git clone https://github.com/YayloyanErik/Voice-Recognition.git
cd Voice-Recognition
pip install -r requirements.txt
```

## Getting started
To start using voice recognition script do following steps:

- Open `model_creation.ipynb` file change names of people(P1_name, P2_name, P3_name) and change P1_voice, P2_voice, P3_voice to your files(suggested length: >5min) and run it

- Open `UI.py`, change names of people(P1_name, P2_name, P3_name) and run

- Toggle the switch and start speaking 

- When switch is off it will show the result
