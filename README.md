## Birdnet- Pi modified 

## Introduction
This version of BirdNET-Pi preserves the complete infrastructure of BirdNET-Pi (recording, real-time analysis, web interface, clip extraction, database, etc.), but replaces the original recognition model with a custom PyTorch ResNet18 model specifically trained for the detection of whale S10 whistles.

The system still runs on a Raspberry Pi, using the power of PyTorch and a ResNet18 model, adapting the BirdNET-Pi workflow for marine bioacoustics.

## Main Differences in This Version
The original BirdNET-Lite (TFLite) model has been replaced by a custom PyTorch (ResNet18) model to identify whale S10 whistles.

The rest of the infrastructure (recording, web interface, databases, clip extraction, etc.) remains unchanged from BirdNET-Pi, allowing for easy installation and use on Raspberry Pi systems.

Ideal for marine monitoring, bioacoustics projects, and automatic detection of whales via the S10 whistle.

## Features
- 24/7 recording and S10 whistle analysis using PyTorch (ResNet18) model

- Automatic extraction of clips when S10 whistles are detected

- Web interface for visualization and downloading of results

- Results stored in a local SQLite3 database

- Live audio streaming

- Spectrogram visualization for detected events

- All the proven and stable BirdNET-Pi infrastructure, now adapted for whales


## Requirements
* A Raspberry Pi 4B or Raspberry Pi 3B+ (The 3B+ must run on RaspiOS-ARM64-**Lite**)
* An SD Card with the **_64-bit version of RaspiOS_** installed (please use Bullseye) -- Lite is recommended, but the installation works on RaspiOS-ARM64-Full as well. [(Download the latest here)](https://downloads.raspberrypi.org/raspios_lite_arm64/images/)
* A USB Microphone or Sound Card

## Installation
[A comprehensive installation guide is available here](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Installation-Guide).

The system can be installed with:
```
curl -s https://raw.githubusercontent.com/dani-cuar/birdnet_pi_modified/stable/newinstaller.sh | bash
```
The installer takes care of any and all necessary updates, so you can run that as the very first command upon the first boot, if you'd like.

## Downloading the Model
Note:
The custom ResNet18 model for whale S10 whistle detection is not included in this repository due to its file size.

To use this system, you must manually download the model and place it in the model/ directory:

Download the model file (resnet18_whales.pth) from this link
https://drive.google.com/file/d/1KzXuzvQvqY4L0hXOec7HAJr0TYzUWqzm/view?usp=drive_link

## Access
The BirdNET-Pi can be accessed from any web browser on the same network:
- http://birdnetpi.local
- Default Basic Authentication Username: birdnet
- Password is empty by default. Set this in "Tools" > "Settings" > "Advanced Settings"

## Uninstallation
```
/usr/local/bin/uninstall.sh && cd ~ && rm -drf BirdNET-Pi
```
