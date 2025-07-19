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

## Access
The BirdNET-Pi can be accessed from any web browser on the same network:
- http://birdnetpi.local
- Default Basic Authentication Username: birdnet
- Password is empty by default. Set this in "Tools" > "Settings" > "Advanced Settings"

Please take a look at the [wiki](https://github.com/mcguirepr89/BirdNET-Pi/wiki) and [discussions](https://github.com/mcguirepr89/BirdNET-Pi/discussions) for information on
- [making your installation public](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Sharing-Your-BirdNET-Pi)
- [backing up and restoring your database](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Backup-and-Restore-the-Database)
- [adjusting your sound card settings](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Adjusting-your-sound-card)
- [suggested USB microphones](https://github.com/mcguirepr89/BirdNET-Pi/discussions/39)
- [building your own microphone](https://github.com/DD4WH/SASS/wiki/Stereo--(Mono)-recording-low-noise-low-cost-system)
- [privacy concerns and options](https://github.com/mcguirepr89/BirdNET-Pi/discussions/166)
- [beta testing](https://github.com/mcguirepr89/BirdNET-Pi/discussions/11)
- [and more!](https://github.com/mcguirepr89/BirdNET-Pi/discussions)

## Uninstallation
```
/usr/local/bin/uninstall.sh && cd ~ && rm -drf BirdNET-Pi
```

## Troubleshooting and Ideas
I want this to work for you! If you have any trouble, or if my documentation is wrong, I'd like to get things right.

If you encounter any issues at any point, or have questions, comments, concerns, ideas, or want to share something, please take a look through the [open and closed issues](https://github.com/mcguirepr89/BirdNET-Pi/issues?q=is%3A+issue) and [the community discussions](https://github.com/mcguirepr89/BirdNET-Pi/discussions). PLEASE feel invited to [open a new issue](https://github.com/mcguirepr89/BirdNET-Pi/issues/new/choose) if you don't find the help you need. Likewise, please accept my invitation to [start a new discussion](https://github.com/mcguirepr89/BirdNET-Pi/discussions/new) to get a conversation started around your topic.

If you are not a GitHub user and need help, you can [email me](mailto:mcguirepr89@gmail.com), but I hope you will consider making a GitHub account so that your questions can be answered here for others as well. I expect this project will attract more bird-enthusiasts than Linux-enthusiasts, so please don't feel like any question is too _novice_ or, pardon the phrase, _stupid_ to ask. I want to help!

## Sharing
Please join a Discussion!! and please join [BirdWeather!!](https://app.birdweather.com)
I hope that if you find BirdNET-Pi has been worth your time, you will share your setup, results, customizations, etc. [HERE](https://github.com/mcguirepr89/BirdNET-Pi/discussions/69) and will consider [making your installation public](https://github.com/mcguirepr89/BirdNET-Pi/wiki/Sharing-Your-BirdNET-Pi).

## ToDo, Notes, and Coming Soon 

### Internationalization:
The bird names are in English by default, but other localized versions are available thanks to the wonderful efforts of [@patlevin](https://github.com/patlevin). Use the web interface's "Tools" > "Settings" and select your "Database Language" to have the detections in your language.

Current database languages include the list below:
| Language | Missing Species out of 6,362 | Missing labels (%) |
| -------- | ------- | ------ |
| Afrikaans | 5774 | 90.76% |
| Catalan | 544 | 8.55% |
| Chinese | 264 | 4.15% |
| Croatian | 370 | 5.82% |
| Czech | 683 | 10.74% |
| Danish | 460 | 7.23% |
| Dutch | 264 | 4.15% |
| Estonian | 3171 | 49.84% |
| Finnish | 518 | 8.14% |
| French | 264 | 4.15% |
| German | 264 | 4.15% |
| Hungarian | 2688 | 42.25% |
| Icelandic | 5588 | 87.83% |
| Indonesian | 5550 | 87.24% |
| Italian | 524 | 8.24% |
| Japanese | 640 | 10.06% |
| Latvian | 4821 | 75.78% |
| Lithuanian | 597 | 9.38% |
| Norwegian | 325 | 5.11% |
| Polish | 265 | 4.17% |
| Portuguese | 2742 | 43.10% |
| Russian | 808 | 12.70% |
| Slovak | 264 | 4.15% |
| Slovenian | 5532 | 86.95% |
| Spanish | 348 | 5.47% |
| Swedish | 264 | 4.15% |
| Thai | 5580 | 87.71% |
| Ukrainian | 646 | 10.15% |

### Tips and Coming Soon:
For some reason, the system seems to run more efficiently and the birds sound better when you [![Star on GitHub](https://img.shields.io/github/stars/mcguirepr89/BirdNET-Pi.svg?style=social)](https://github.com/mcguirepr89/BirdNET-Pi/stargazers) and <a href="https://github.com/sponsors/mcguirepr89">Sponsor</a> the project :)

Expect FULL internationalization options post-installation for the following languages:
- German
- Swedish
- French
- Spanish
