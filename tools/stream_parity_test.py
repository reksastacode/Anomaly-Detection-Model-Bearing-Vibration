"""
ZENTRA TinyML - Hardware Parity Streamer & Serial Monitor (Tahap 12)
Listens to ESP32-S3 serial stream, captures embedded parity reports,
and validates tolerances between Python and C firmware.

Usage:
  python tools/stream_parity_test.py --port COM3 --baud 115200
  python tools/stream_parity_test.py --scan
"""

import sys
import time
import argparse
import serial
import serial.tools.list_ports

def scan_ports():
    ports = list(serial.tools.list_ports.comports())
    print("\n--- Available Serial (COM) Ports ---")
    if not ports:
        print("  No serial devices detected. Check USB connection.")
    for p in ports:
        print(f"  {p.device} - {p.description} [{p.hwid}]")
    print("------------------------------------\n")
    return [p.device for p in ports]

def monitor_parity_test(port_name, baud=115200, timeout=15):
    print(f"\n[PARITY MONITOR] Opening serial port {port_name} at {baud} baud...")
    try:
        ser = serial.Serial(port_name, baud, timeout=2)
    except Exception as e:
        print(f"[ERROR] Failed to open serial port {port_name}: {e}")
        return False
        
    print("[PARITY MONITOR] Port open. Waiting for ESP32-S3 reboot / parity test output...\n")
    time.sleep(1.0)
    # Trigger soft-reset via DTR/RTS if supported (typical for ESP32 DevKit)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setDTR(True)
    ser.setRTS(False)
    
    start_time = time.time()
    parity_passed = False
    
    try:
        while time.time() - start_time < timeout:
            line = ser.readline().decode('utf-8', errors='replace').strip()
            if line:
                print(f"[ESP32] {line}")
                if "ALL PARITY TESTS PASSED" in line:
                    parity_passed = True
                elif "PARITY FAILED" in line:
                    parity_passed = False
                    
            if parity_passed and "FreeRTOS Subsystem aktif" in line:
                break
    except KeyboardInterrupt:
        print("\n[PARITY MONITOR] Monitoring terminated by user.")
    finally:
        ser.close()
        
    print("\n" + "=" * 60)
    if parity_passed:
        print(">>> RESULT: PARITY VERIFICATION PASSED (Hardware matches Python) <<<")
    else:
        print(">>> RESULT: PARITY PENDING / TIMEOUT (Re-check ESP32 boot output) <<<")
    print("=" * 60 + "\n")
    return parity_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZENTRA Hardware Parity Monitor")
    parser.add_argument("--port", type=str, default=None, help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--scan", action="store_true", help="Scan and list available ports")
    args = parser.parse_args()
    
    if args.scan:
        scan_ports()
    elif args.port:
        monitor_parity_test(args.port, args.baud)
    else:
        available = scan_ports()
        if available:
            print(f"Auto-selecting first port: {available[0]}")
            monitor_parity_test(available[0], args.baud)
        else:
            print("[INFO] No ESP32-S3 port detected. Connect device via USB and run with --port COMx")
