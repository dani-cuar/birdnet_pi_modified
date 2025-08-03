#!/home/pi/BirdNET-Pi/birdnet/bin/python3
import socket 
import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

try:
    import tflite_runtime.interpreter as tflite
except:
    from tensorflow import lite as tflite

import argparse
import operator
import librosa
import numpy as np
import math
import time
from decimal import Decimal
import json
import requests
import sqlite3
import datetime
from time import sleep
import pytz
from tzlocal import get_localzone
from pathlib import Path

########## AGREGADO #############
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import models
import logging
import sys

from collections import deque

S10_LABEL = "S10"
S10_THRESHOLD = 0.8         # Score mínimo para considerar S10 (ajústalo si hace falta)
WINDOW_SECONDS = 120        # 2 minutos
S10_COUNT_TRIGGER = 10      # Número de S10 para alertar

# Variables de estado
global s10_count_window, window_start_time
s10_count_window = 0
window_start_time = time.time()

# from sim800c.sim800c import Sim800C


# gsm = Sim800C(serial_port="/dev/ttyUSB0", baudrate=9600, bootup=True)
#################################

HEADER = 64
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    server.bind(ADDR)
except:
    print("Waiting on socket")
    time.sleep(5)
    
# =============== CONFIG LOGGING ====================
log_dir = '/home/pi/BirdNET-Pi/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# # Configurar el archivo de log
log_filename = os.path.join(log_dir, 'server_alert_log.txt')

# Configuración de logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',         # <--- clave
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename),
        # logging.StreamHandler()
    ]
)


# Crea un StreamHandler para que los logs se impriman en la consola
console_handler = logging.StreamHandler()
logging.getLogger().addHandler(console_handler)
# # ====================================================

# Open most recent Configuration and grab DB_PWD as a python variable
userDir = os.path.expanduser('~')
with open(userDir + '/BirdNET-Pi/scripts/thisrun.txt', 'r') as f:
    this_run = f.readlines()
    audiofmt = "." + str(str(str([i for i in this_run if i.startswith('AUDIOFMT')]).split('=')[1]).split('\\')[0])

############### AGREGADO ####################################
def build_model(num_classes: int, remove_initial_maxpool: bool = True):
    m = models.resnet18(pretrained=False)
    if remove_initial_maxpool:
        m.maxpool = torch.nn.Identity()
    in_f = m.fc.in_features
    m.fc = torch.nn.Linear(in_f, num_classes)
    return m
#############################################################

def loadModel():

    ########## ORIGINAL ##########
    # global INPUT_LAYER_INDEX
    # global OUTPUT_LAYER_INDEX
    # global MDATA_INPUT_INDEX
    global CLASSES
    ##############################
    global model # agregado

    # logging.info('LOADING TF LITE MODEL...')

    ################# ORIGINAL #######################

    # Load TFLite model and allocate tensors.
    # modelpath = userDir + '/BirdNET-Pi/model/BirdNET_6K_GLOBAL_MODEL.tflite'
    # myinterpreter = tflite.Interpreter(model_path=modelpath,num_threads=1)
    # myinterpreter.allocate_tensors()

    # # Get input and output tensors.
    # input_details = myinterpreter.get_input_details()
    # output_details = myinterpreter.get_output_details()

    # # Get input tensor index
    # INPUT_LAYER_INDEX = input_details[0]['index']
    # MDATA_INPUT_INDEX = input_details[1]['index']
    # OUTPUT_LAYER_INDEX = output_details[0]['index']

    # # Load labels
    # CLASSES = []
    # labelspath = userDir + '/BirdNET-Pi/model/labels.txt'
    # with open(labelspath, 'r') as lfile:
    #     for line in lfile.readlines():
    #         CLASSES.append(line.replace('\n', ''))

    # print('DONE!')

    # return myinterpreter

    ################## AGREGADO ################################
    modelpath = userDir + '/BirdNET-Pi/model/resnet18_whales.pth' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(2, remove_initial_maxpool=True)  # Aquí 2 es el número de clases
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.to(device).eval()  # Colocar el modelo en modo de inferencia

    # Cargar las etiquetas (si están en un archivo de texto similar al original)
    CLASSES = []
    with open(userDir + '/BirdNET-Pi/model/labels_whale.txt', 'r') as lfile:
        for line in lfile.readlines():
            CLASSES.append(line.strip())  # Elimina saltos de línea

    # logging.info('¡Modelo cargado con éxito!')

    return model
    ###########################################################

def loadCustomSpeciesList(path):

    slist = []
    if os.path.isfile(path):
        with open(path, 'r') as csfile:
            for line in csfile.readlines():
                slist.append(line.replace('\r', '').replace('\n', ''))

    return slist

def splitSignal(sig, rate, overlap, seconds=3.0, minlen=1.5):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short? Fill with zeros.
        if len(split) < int(rate * seconds):
            temp = np.zeros((int(rate * seconds)))
            temp[:len(split)] = split
            split = temp
        
        sig_splits.append(split)

    return sig_splits

################## AGREGADO #########################################

class AudioPreprocessor:
    def __init__(self, fs=6250, n_fft=512, overlap=0.75, fmin=300, fmax=600):
        self.fs = fs
        self.n_fft = n_fft
        self.hop_len = int(n_fft * (1 - overlap))
        self.window = torch.hann_window(n_fft)
        freqs = torch.fft.rfftfreq(n_fft, d=1/fs)
        self.freq_mask = (freqs >= fmin) & (freqs <= fmax)

    def __call__(self, wav_t: torch.Tensor) -> torch.Tensor:
        # wav_t: 1-D tensor a fs de self.fs
        stft = torch.stft(wav_t, n_fft=self.n_fft, hop_length=self.hop_len,
                          window=self.window, return_complex=True)
        # spec = torch.abs(stft)[self.freq_mask]  # (n_bins, n_frames)
        spec = torch.abs(stft)  # (1, 257, N)
        spec = spec[:, self.freq_mask, :]  # Mantén batch o canal en el eje 0
        med = torch.median(spec, dim=-1, keepdim=True)[0]
        spec = spec / (med + 1e-8)
        mn, mx = spec.min(), spec.max()
        spec = (spec - mn) / (mx - mn + 1e-8)
        return spec.unsqueeze(0)  # shape (1, n_bins, n_frames)
#####################################################################

def readAudioData(path, overlap, sample_rate=6250): #sr original 48000

    # logging.info('READING AUDIO DATA...')

    ############# ORIGINAL #######################
    # Open file with librosa (uses ffmpeg or libav)
    # sig, rate = librosa.load(path, sr=sample_rate, mono=True, res_type='kaiser_fast')

    # # Split audio into 3-second chunks
    # chunks = splitSignal(sig, rate, overlap)

    # print('DONE! READ', str(len(chunks)), 'CHUNKS.')

    # return chunks
    ##############################################

    ############# AGREGADO ##############################
    # Cargar el archivo de audio
    sig, rate = torchaudio.load(path)

    # --- AQUI: Convertir a mono si tiene más de un canal ---
    if sig.size(0) > 1:
        sig = sig.mean(dim=0, keepdim=True)  # shape [1, muestras]
    # ------------------------------------------------------

    # Remuestrear el audio si la frecuencia de muestreo es diferente a la esperada
    if rate != sample_rate:
        sig = torchaudio.transforms.Resample(rate, sample_rate)(sig)

    # Dividir el audio en fragmentos de 2 segundos
    seg_len = sample_rate * 2  # 2 segundos
    n_segs = math.ceil(sig.size(1) / seg_len)

    prep = AudioPreprocessor()
    chunks = []
    for i in range(n_segs):
        start = i * seg_len
        end = start + seg_len
        seg = sig[:, start:end]

        # Rellenar con ceros si el último fragmento es más corto
        if seg.size(1) < seg_len:
            pad_amt = seg_len - seg.size(1)
            seg = F.pad(seg, (0, pad_amt))

        # Preparar el espectrograma
        spec = prep(seg)  # (1, n_bins, n_frames)
        chunks.append(spec)

    # logging.info(f'Read {len(chunks)} chunks from the audio file.')
    return chunks
    ######################################################

def convertMetadata(m):

    # Convert week to cosine
    if m[2] >= 1 and m[2] <= 48:
        m[2] = math.cos(math.radians(m[2] * 7.5)) + 1 
    else:
        m[2] = -1

    # Add binary mask
    mask = np.ones((3,))
    if m[0] == -1 or m[1] == -1:
        mask = np.zeros((3,))
    if m[2] == -1:
        mask[2] = 0.0

    return np.concatenate([m, mask])

def custom_sigmoid(x, sensitivity=1.0):
    return 1 / (1.0 + np.exp(-sensitivity * x))

def predict(sample, sensitivity):

    ############### ORIGINAL ############################
    # global INTERPRETER
    # # Make a prediction
    # INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample[0], dtype='float32'))
    # INTERPRETER.set_tensor(MDATA_INPUT_INDEX, np.array(sample[1], dtype='float32'))
    # INTERPRETER.invoke()
    # prediction = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)[0]

    # # Apply custom sigmoid
    # p_sigmoid = custom_sigmoid(prediction, sensitivity)

    # # Get label and scores for pooled predictions
    # p_labels = dict(zip(CLASSES, p_sigmoid))

    # # Sort by score
    # p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

    # # Remove species that are on blacklist
    # for i in range(min(10, len(p_sorted))):
    #     if p_sorted[i][0] in ['Human_Human', 'Non-bird_Non-bird', 'Noise_Noise']:
    #         p_sorted[i] = (p_sorted[i][0], 0.0)

    # # Only return first the top ten results
    # return p_sorted[:10]
    ##########################################################

    ####################### AGREGADO #########################
    global model  # Usamos el modelo cargado
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Preprocesar y enviar al modelo
    audio_chunk = sample[0].to(device)  # Pasamos el fragmento de audio al dispositivo adecuado

    # === Medir tiempo de inferencia ===
    # start_infer = time.perf_counter()  # Inicio
    with torch.no_grad():
        output = model(audio_chunk)  # Realizamos la inferencia
        probs = F.softmax(output, dim=1)
        idx = output.argmax(dim=1).item()  # Obtenemos la clase con mayor probabilidad

    # end_infer = time.perf_counter()    # Fin
    # infer_time = end_infer - start_infer

    # === Guardar el tiempo en un log ===
    # with open('/home/pi/BirdNET-Pi/infer_times.log', 'a') as flog:
    #     flog.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}, inference_time={infer_time:.4f} seconds\n")

    # Mapeo de la predicción a las etiquetas 
    predicted_label = CLASSES[idx]  # Usamos el índice de la predicción para obtener la clase correspondiente
    score = probs[0][idx].item()
    # Ordenar las predicciones (en este caso solo hay una predicción, pero si fuese necesario)
    # p_labels = {predicted_label: output[0][idx].item()}  # Guardamos la clase y la probabilidad
    p_labels = {predicted_label: score}

    return p_labels  # Solo devolvemos la clase y la probabilidad

def analyzeAudioData(chunks, lat, lon, week, sensitivity, overlap,):
    # global INTERPRETER #ORIGINAL
    #### AGREGADO ####
    global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##################

    detections = {}
    start = time.time()
    # logging.info('ANALYZING AUDIO...')

    # Convert and prepare metadata
    mdata = convertMetadata(np.array([lat, lon, week]))
    mdata = np.expand_dims(mdata, 0)

    # Parse every chunk
    pred_start = 0.0
    for c in chunks:

        # Prepare as input signal
        ######## ORIGINAL #########
        # sig = np.expand_dims(c, 0)

        # # Make prediction
        # p = predict([sig, mdata], sensitivity)
        ###########################

        ######### AGREGADO #############
        sig = c.repeat(1, 3, 1, 1) # c= (1, n_bins, n_frames)

        sig = F.interpolate(sig, size=(224,224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
        std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
        sig = (sig.to(device) - mean) / std
        p = predict([sig, mdata], sensitivity)
        ################################

        # Save result and timestamp
        # pred_end = pred_start + 3.0

        ######## AGREGADO ###############
        pred_end = pred_start + 2.0  # Duración de cada fragmento es de 2 segundos
        #################################

        detections[str(pred_start) + ';' + str(pred_end)] = p
        pred_start = pred_end - overlap

    # logging.info(f'Analysis done in {int((time.time() - start) * 10) / 10.0} SECONDS.')

    return detections

def writeResultsToFile(detections, min_conf, path):

    # logging.info(f'WRITING RESULTS TO {path}...')
    rcnt = 0
    with open(path, 'w') as rfile:
        # rfile.write('Start (s);End (s);Scientific name;Common name;Confidence\n')
        rfile.write("Start (s);End (s);Sci_Name;Com_Name;Confidence\n")
        for d in detections:
            for label, score in detections[d].items():
                if score >= min_conf and ((label in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (label not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)):
                    # rfile.write(f"{d};{sci};{com};{score}\n")
                    rfile.write(f"{d};{label};{label};{score}\n")
                    rcnt += 1
            # for entry in detections[d]:
            #     if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0) ):
            #         rfile.write(d + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) + '\n')
            #         rcnt += 1
    # logging.info(f'DONE! WROTE {rcnt} RESULTS.')
    return

# Estado persistente del sistema de alerta
audio_clock = 0.0
window_start_time = 0.0
s10_count_window = 0

def handle_client(conn, addr):
    global s10_count_window, window_start_time, audio_clock
    # Crea un logger específico para este hilo
    logger = logging.getLogger(f"Client-{addr}")
    # logger.info(f"New connection from {addr}")
    global INCLUDE_LIST
    global EXCLUDE_LIST
    # print(f"[NEW CONNECTION] {addr} connected.")

    # connected = True
    # while connected:
    myReturn = "NO_RESULTS"
    try:
        # logger.info("Recibiendo datos del cliente...")
        msg_length = conn.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = conn.recv(msg_length).decode(FORMAT)
            if msg == DISCONNECT_MESSAGE:
                # connected = False
                # logger.info("Received disconnect message.")
                myReturn = "DISCONNECTED"
            else:
                # logger.info(f"Message received: {msg}")
                #print(f"[{addr}] {msg}")
                
                args = type('', (), {})()
                
                args.i = ''
                args.o = ''
                args.birdweather_id = '99999'
                args.include_list = 'null'
                args.exclude_list = 'null'
                args.overlap = 0.0
                args.week = -1
                args.sensitivity = 1.25
                args.min_conf = 0.70
                args.lat = -1
                args.lon =  -1


                for line in msg.split('||'):
                    inputvars = line.split('=')
                    if inputvars[0] == 'i':
                        args.i = inputvars[1]
                    elif inputvars[0] == 'o':
                        args.o = inputvars[1]
                    elif inputvars[0] == 'birdweather_id':
                        args.birdweather_id = inputvars[1]
                    elif inputvars[0] == 'include_list':
                        args.include_list = inputvars[1]
                    elif inputvars[0] == 'exclude_list':
                        args.exclude_list = inputvars[1]
                    elif inputvars[0] == 'overlap':
                        args.overlap = float(inputvars[1])
                    elif inputvars[0] == 'week':
                        args.week = int(inputvars[1])
                    elif inputvars[0] == 'sensitivity':
                        args.sensitivity = float(inputvars[1])
                    elif inputvars[0] == 'min_conf':
                        args.min_conf = float(inputvars[1])
                    elif inputvars[0] == 'lat':
                        args.lat = float(inputvars[1])
                    elif inputvars[0] == 'lon':
                        args.lon = float(inputvars[1])


                
                # Load custom species lists - INCLUDED and EXCLUDED
                if not args.include_list == 'null':
                    INCLUDE_LIST = loadCustomSpeciesList(args.include_list)
                else:
                    INCLUDE_LIST = []
                
                if not args.exclude_list == 'null':
                    EXCLUDE_LIST = loadCustomSpeciesList(args.exclude_list)
                else:
                    EXCLUDE_LIST = []

                birdweather_id = args.birdweather_id

                # Read audio data
                audioData = readAudioData(args.i, args.overlap)

                # Get Date/Time from filename in case Pi gets behind
                #now = datetime.now()
                full_file_name = args.i
                print('FULL FILENAME: -' + full_file_name + '-')
                file_name = Path(full_file_name).stem
                file_date = file_name.split('-birdnet-')[0]
                file_time = file_name.split('-birdnet-')[1]
                date_time_str = file_date + ' ' + file_time
                date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                #print('Date:', date_time_obj.date())
                #print('Time:', date_time_obj.time())
                print('Date-time:', date_time_obj)
                now = date_time_obj
                current_date = now.strftime("%Y-%m-%d")
                current_time = now.strftime("%H:%M:%S")
                current_iso8601 = now.astimezone(get_localzone()).isoformat()
                
                week_number = int(now.strftime("%V"))
                week = max(1, min(week_number, 48))

                sensitivity = max(0.5, min(1.0 - (args.sensitivity - 1.0), 1.5))

                # Process audio data and get detections
                detections = analyzeAudioData(audioData, args.lat, args.lon, week, sensitivity, args.overlap)

                # Write detections to output file
                min_conf = max(0.01, min(args.min_conf, 0.99))
                writeResultsToFile(detections, min_conf, args.o)
                
            ###############################################################################    
            ###############################################################################    
                
                soundscape_uploaded = False
                
                # Write detections to Database
                myReturn = ''
                
                for i in detections:
                    for label, score in detections[i].items():
                        myReturn += f"{i}-{label}-{score}\n"
                #   myReturn += str(i) + '-' + str(detections[i][0]) + '\n'

                # with open(userDir + '/BirdNET-Pi/BirdDB.txt', 'a') as rfile:
                with open(userDir + '/BirdNET-Pi/detections_whale.txt', 'a') as rfile:
                    for d in detections:
                        start_sec = float(d.split(';')[0])
                        end_sec = float(d.split(';')[1])

                        for label, score in detections[d].items():
                            if score >= min_conf and ((label in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (label not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0)):
                                # rfile.write(str(current_date) + ';' + str(current_time) + ';' + label.replace('_', ';') + ';' \
                                # + str(score) + ";" + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';' \
                                # + str(args.sensitivity) + ';' + str(args.overlap) + '\n')
                                rfile.write(f"{current_date};{current_time};{label};{score};{args.lat};{args.lon};{min_conf};{week};{args.sensitivity};{args.overlap}\n")
                        # for entry in detections[d]:
                        #     if entry[1] >= min_conf and ((entry[0] in INCLUDE_LIST or len(INCLUDE_LIST) == 0) and (entry[0] not in EXCLUDE_LIST or len(EXCLUDE_LIST) == 0) ):
                        #         rfile.write(str(current_date) + ';' + str(current_time) + ';' + entry[0].replace('_', ';') + ';' \
                        #         + str(entry[1]) +";" + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';' \
                        #         + str(args.sensitivity) +';' + str(args.overlap) + '\n')
                                
                                Date = str(current_date)
                                Time = str(current_time)
                                # species = entry[0]
                                species = label
                                if '_' in species:
                                    Sci_Name, Com_Name = species.split('_', 1)
                                else:
                                    Sci_Name = species
                                    Com_Name = species
                                # score = entry[1]
                                Confidence = str(round(score*100))
                                Lat = str(args.lat)
                                Lon = str(args.lon)
                                Cutoff = str(args.min_conf)
                                Week = str(args.week)
                                Sens = str(args.sensitivity)
                                Overlap = str(args.overlap)
                                # Com_Name = Com_Name.replace("'", "")
                                File_Name = label.replace(" ", "_") + '-' + Confidence + '-' + Date.replace("/", "-") + '-birdnet-' + Time + audiofmt

                                # File_Name = Com_Name.replace(" ", "_") + '-' + Confidence + '-' + \
                                #         Date.replace("/", "-") + '-birdnet-' + Time + audiofmt

                                #Connect to SQLite Database
                                # con = sqlite3.connect(userDir + '/BirdNET-Pi/scripts/birds.db')
                                con = sqlite3.connect(userDir + '/BirdNET-Pi/scripts/detections_whale.db')
                                cur = con.cursor()

                                # Crea la tabla si no existe
                                cur.execute("""
                                    CREATE TABLE IF NOT EXISTS detections (
                                        Date TEXT,
                                        Time TEXT,
                                        Sci_Name TEXT,
                                        Com_Name TEXT,
                                        Score TEXT,
                                        Lat TEXT,
                                        Lon TEXT,
                                        Cutoff TEXT,
                                        Week TEXT,
                                        Sens TEXT,
                                        Overlap TEXT,
                                        File_Name TEXT
                                    )
                                """)
                                # Usa label para ambas columnas
                                Sci_Name = label
                                Com_Name = label
                                cur.execute("INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (Date, Time, Sci_Name, Com_Name, str(score), Lat, Lon, Cutoff, Week, Sens, Overlap, File_Name))

                                con.commit()
                                con.close()

                                # ---- Lógica de alerta S10 ----
                                if label == S10_LABEL and score >= S10_THRESHOLD:
                                    # now_time = time.time()

                                    # Incrementar contador de llamados S10 en esta ventana
                                    s10_count_window += 1
                                    # logging.info(f"ALERTA: {s10_count_window} detectado")

                        # Actualizar el reloj del audio
                        audio_clock += end_sec
                        # logging.info(f"Audio clock: {audio_clock}")
                        logging.info(f"Tiempo acumu: {audio_clock - window_start_time:.2f}s | S10: {s10_count_window}")
                        # Si pasaron 2 minutos (120 segundos) desde el inicio de la ventana
                        if audio_clock - window_start_time >= WINDOW_SECONDS:
                            if s10_count_window >= S10_COUNT_TRIGGER:
                                logging.info(f"ALERTA: {s10_count_window} S10 detectados entre segundo {window_start_time} y {audio_clock}")
                                try:
                                    # gsm.send_sms("+573001234567", "10 S10 detectados")
                                    logging.info("SMS de alerta enviado correctamente")
                                except Exception as e:
                                    logging.error(f"Error al enviar SMS de alerta: {e}")
                            else:
                                logging.info(f"NO ALERTA: Solo {s10_count_window} S10 detectados entre segundo {window_start_time} y {audio_clock}")

                            # Reiniciar ventana
                            s10_count_window = 0
                            window_start_time = audio_clock

                                # print(str(current_date) + ';' + str(current_time) + ';' + entry[0].replace('_', ';') + ';' + str(entry[1]) + ';' + str(args.lat) + ';' + str(args.lon) + ';' + str(min_conf) + ';' + str(week) + ';' + str(args.sensitivity) +';' + str(args.overlap) + Com_Name.replace(" ", "_") + '-' + str(score) + '-' + str(current_date) + '-birdnet-' + str(current_time) + audiofmt  + '\n')
                                # Print con solo label y score
                                # print(f"{current_date};{current_time};{label};{score};{args.lat};{args.lon};{min_conf};{week};{args.sensitivity};{args.overlap};{Com_Name.replace(' ', '_')}-{score}-{current_date}-birdnet-{current_time}{audiofmt}\n")

                #                 if birdweather_id != "99999":
                #                     try:

                #                         if soundscape_uploaded is False:
                #                             # POST soundscape to server
                #                             soundscape_url = "https://app.birdweather.com/api/v1/stations/" + birdweather_id +  "/soundscapes" + "?timestamp=" + current_iso8601
    
                #                             with open(args.i, 'rb') as f:
                #                                 wav_data = f.read()
                #                             response = requests.post(url=soundscape_url, data=wav_data, headers={'Content-Type': 'application/octet-stream'})
                #                             # print("Soundscape POST Response Status - ", response.status_code)
                #                             sdata = response.json()
                #                             soundscape_id = sdata['soundscape']['id']
                #                             soundscape_uploaded = True
    
                #                         # POST detection to server
                #                         detection_url = "https://app.birdweather.com/api/v1/stations/" + birdweather_id + "/detections"
                #                         start_time = d.split(';')[0]
                #                         end_time = d.split(';')[1]
                #                         post_begin = "{ "
                #                         now_p_start = now + datetime.timedelta(seconds=float(start_time))
                #                         current_iso8601 = now_p_start.astimezone(get_localzone()).isoformat()
                #                         post_timestamp =  "\"timestamp\": \"" + current_iso8601 + "\","
                #                         post_lat = "\"lat\": " + str(args.lat) + ","
                #                         post_lon = "\"lon\": " + str(args.lon) + ","
                #                         post_soundscape_id = "\"soundscapeId\": " + str(soundscape_id) + ","
                #                         post_soundscape_start_time = "\"soundscapeStartTime\": " + start_time + ","
                #                         post_soundscape_end_time = "\"soundscapeEndTime\": " + end_time + ","
                #                         # post_commonName = "\"commonName\": \"" + entry[0].split('_')[1] + "\","
                #                         # post_scientificName = "\"scientificName\": \"" + entry[0].split('_')[0] + "\","
                #                         post_commonName = f"\"commonName\": \"{label}\","
                #                         post_scientificName = f"\"scientificName\": \"{label}\","
                #                         post_algorithm = "\"algorithm\": " + "\"alpha\"" + ","
                #                         # post_confidence = "\"confidence\": " + str(entry[1])
                #                         post_confidence = "\"confidence\": " + str(score)
                #                         post_end = " }"
    
                #                         post_json = post_begin + post_timestamp + post_lat + post_lon + post_soundscape_id + post_soundscape_start_time + post_soundscape_end_time + post_commonName + post_scientificName + post_algorithm + post_confidence + post_end
                #                         # print(post_json)
                #                         response = requests.post(detection_url, json=json.loads(post_json))
                #                         # print("Detection POST Response Status - ", response.status_code)
                #                     except:
                #                         print("Cannot POST right now")
                conn.send(myReturn.encode(FORMAT))

                                #time.sleep(3)
    except Exception as e:
        logger.error(f"Error while processing client: {e}")
        # Siempre responde algo aunque falle
        try:
            conn.send(f"ERROR: {str(e)}".encode(FORMAT))
        except Exception as inner:
            logger.error(f"Error sending error response: {inner}")
    finally:
        conn.close()
        # logger.info("Connection closed.")

    # conn.close() 

def start():
    # Load model
    # global INTERPRETER, #ORIGINAL
    global INCLUDE_LIST, EXCLUDE_LIST
    ### AGREGADO ####
    global model
    #################

    #### ORIGINAL ######
    # INTERPRETER = loadModel()
    ####################

    ####### AGREGADO ####
    model = loadModel()
    #####################
    
    server.listen()
    # logging.info(f"[LISTENING] Server is listening on {SERVER}")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        # logging.info(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")


# logging.info("[STARTING] server is starting...")
start()