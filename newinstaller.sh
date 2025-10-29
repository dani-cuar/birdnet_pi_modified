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

# Clonar el repo modificado
git clone -b $branch https://github.com/dani-cuar/birdnet_pi_modified.git ${HOME}/BirdNET-Pi &&

# ===== DESCARGA AUTOMÁTICA DEL MODELO DESDE GOOGLE DRIVE =====
DEST_DIR="${HOME}/BirdNET-Pi/model"
mkdir -p "$DEST_DIR"

FILE_ID="113V5ecW-iTl1-N28UqrQK3jNgk9BH0qW"

echo "Descargando modelo resnet18_whales.pth desde Google Drive..."
curl -fL --retry 3 --retry-delay 2 \
  -o "${DEST_DIR}/resnet18_whales.pth" \
  "https://drive.google.com/uc?export=download&id=${FILE_ID}"

# Verificar la descarga
if [ ! -s "${DEST_DIR}/resnet18_whales.pth" ]; then
  echo "Error: no se pudo descargar el modelo desde Google Drive."
  exit 1
else
  echo "Modelo descargado correctamente en ${DEST_DIR}/resnet18_whales.pth"
fi
# ===== FIN DE DESCARGA =====

# Ejecutar el instalador principal
"${HOME}/BirdNET-Pi/scripts/install_birdnet.sh"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
  echo "Installation completed successfully"
  sudo reboot
else
  echo "The installation exited unsuccessfully."
  exit 1
fi



