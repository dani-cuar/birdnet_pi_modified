import logging, time
from sim800c.sim800c import Sim800C

def main():
    # 1) Logging del script
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")
    # Asegúrate también de tener LOG_LEVEL="DEBUG" en sim800c/config.py

    sim = Sim800C(bootup=False, serial_port="/dev/ttyUSB0")

    # 2) Abre el puerto
    sim.init_serial()

    # 3) Prueba AT y muestra salida
    res = sim.do_at_command('AT', pause_time=1.0)
    print("AT =>", repr(res))  # Deberías ver '\r\nOK\r\n'

    # 4) Checa PIN y muestra salida
    res = sim.do_at_command('AT+CPIN?', pause_time=1.0)
    print("AT+CPIN? =>", repr(res))  # Esperado: '+CPIN: READY'

    # 5) (Opcional) Forzar modo texto y mostrar
    res = sim.do_at_command('AT+CMGF=1', pause_time=1.2)
    print("AT+CMGF=1 =>", repr(res))  # Esperado: 'OK'

    # 6) Enviar SMS con throttle bajo y confirmar envío
    sim.sms_throttle_time = 5

    # --- versión con confirmación del módem ---
    # CMGS/OK check (sin tocar la librería):
    #   a) prepara número -> espera '>'
    cmd = sim._at_commands('set_sms_phone_number') + f"\"{'+526121270531'}\""
    res = sim.do_at_command(cmd, pause_time=1.2)
    print("AT+CMGS =>", repr(res))
    if '>' not in res:
        raise RuntimeError("El módem no dio prompt '>' para el texto del SMS.")

    #   b) envía texto + Ctrl+Z, lee confirmación
    res = sim.do_at_command("hola" + "\x1a", pause_time=3.0)
    time.sleep(2)  # da tiempo a que termine
    try:
        res += sim.ser.read(sim.ser.in_waiting).decode(errors='ignore')
    except Exception:
        pass
    print("SMS respuesta final =>", repr(res))
    if "+CMGS" in res and "OK" in res:
        print("SMS enviado (confirmado por el módem).")
    else:
        print("El módem NO confirmó el envío:", res)

    # --- si prefieres tu función directa (no confirma CMGS/OK) ---
    # sim.sms.send_sms_message("+526121270531", "hola")
    # print("SMS enviado (sin confirmación CMGS/OK)")

if __name__ == "__main__":
    main()
