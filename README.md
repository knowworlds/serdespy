# serdespy
A python library for system-level SerDes modelling and simulation

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Setup](#setup)
* [Examples](#examples)
* [112G PAM4 link eye](#112g-pam4-link-eye)
* [Project Status](#project-status)

## General Information

Richard Barrie's undergraduate thesis project. University of Toronto, Engineering Science. ESC499 2021/2022. 
Supervisors: Tony Chan Causone, Ming Yang
Authors: Richard Barrie, Katherine Liang

## Features
Includes functions and classes for time-domain model of serdes system
- Channel Modelling
- TX FIR Filter
- TX Jitter
- Continuous-Time Linear Equalizer (CTLE)
- Feed-Forward Equalizer (FFE)
- Decision Feedback Equalizer (DFE)
- Maximum-Likelihood Sequence Estimation (MLSE)
- PRBS/PRQS generation
- Eye Diagram Plotter
- Bit Error Rate Checker
- Forward Error Correction with Reed-Solomon Codes

## Setup
Python 3.7+ required
`pip install serdespy`

## Examples
The 'examples' directory contains some scripts and documentation on how to use serdespy. It contains 2 subdirectories:

- /problem_set
    - contains a problem set on wireline links "problem_set.pdf", and scripts that complete the problems using the serdespy package
    - files should be run in numerical order

- /example_channel
    - this directory contains a series of scripts that perform modelling and simulation on a model of a 100G PAM-4 copper channel from 4-port s-paramaters
    - instructions for downloading the s-paramater files and running the scripts are in the header of "1_channel.py"
    - files should be run in numerical order

## 112G PAM4 link eye

`src/112g_pam4/` is a NumPy port of the signal chain behind the
[112G PAM4 SerDes Link](https://circuits-and-systems.tokenzhang.com/serdes/112g-pam4-link/)
animation: TX FFE, skin/dielectric channel, CTLE, VGA, 12-tap MMSE FFE and optional 1-tap DFE.
It is separate from the `serdespy` package.

Requires NumPy and Matplotlib (already listed in `setup.py`):

```bash
python src/112g_pam4/plot_eye.py
```

That solves the page's default link (28 dB loss, TX FFE on, CTLE auto, FFE, 1.5 mV crosstalk, 0.8 mV RX noise) and writes `src/112g_pam4/eye_default.png`: the closed pad eye, the 2 UI sampling eye after CTLE + FFE, and the centre-slice histogram. On the default settings the script also checks the solved cursors and SNR against the digits the animation publishes. Other controls match the page sliders:

```bash
python src/112g_pam4/plot_eye.py --loss 42 --dsp dfe --xt-mv 2.1 --rx-noise-mv 0.6 -o eye_long.png
python src/112g_pam4/plot_eye.py --loss 12 --echo 0.85 --dsp dfe -o eye_echo.png
```
