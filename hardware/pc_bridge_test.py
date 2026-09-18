"""
Teste manual do protocolo serial PC<->placa, antes de plugar o cerebro de verdade.
Funciona tanto com Wokwi (porta serial virtual exposta pela extensao VS Code)
quanto com Arduino real depois.

Uso: python pc_bridge_test.py COM5
Aperta Enter no terminal a qualquer momento -> manda "E" (escape) pra placa.
"""
import sys
import threading
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "COM5"
BAUD = 115200


def read_loop(ser):
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if line:
            print(f"[placa] {line}")


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f"Conectado em {PORT} @ {BAUD}. Enter = manda escape ('E'). Ctrl+C sai.")

    t = threading.Thread(target=read_loop, args=(ser,), daemon=True)
    t.start()

    try:
        while True:
            input()
            ser.write(b"E")
            print("[pc] enviado E (escape)")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
