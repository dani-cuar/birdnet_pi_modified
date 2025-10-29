#!/usr/bin/env bash
# Simple new installer
HOME=$HOME
USER=$USER

export HOME=$HOME
export USER=$USER

# branch=main ##ORIGINAL
branch=stable
if ! which git &> /dev/null; then
  sudo apt update
  sudo apt -y install git
fi

git clone -b $branch https://github.com/dani-cuar/birdnet_pi_modified.git ${HOME}/BirdNET-Pi &&
# Ejecutar el instalador principal
"${HOME}/BirdNET-Pi/scripts/install_birdnet.sh"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "Installation completed successfully"
  sudo reboot
else
  echo "The installation exited unsuccessfully."
  exit 1
fi



