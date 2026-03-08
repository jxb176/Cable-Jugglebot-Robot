#!/usr/bin/env python3
"""
Simple test client for the Cable-Jugglebot simulation.
Connects to the TCP command server and sends test commands.
"""

import socket
import json
import time
import sys

TCP_HOST = "localhost"
TCP_PORT = 5555

def send_command(sock, cmd):
    """Send a JSON command to the server."""
    msg = json.dumps(cmd) + "\n"
    sock.sendall(msg.encode("utf-8"))
    print(f"Sent: {msg.strip()}")

def main():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((TCP_HOST, TCP_PORT))
        print(f"Connected to {TCP_HOST}:{TCP_PORT}")

        # Wait a bit for connection
        time.sleep(1)

        # Enable the robot
        send_command(sock, {"type": "state", "value": "enable"})
        time.sleep(2)

        # Set home position
        send_command(sock, {"type": "home", "home_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
        time.sleep(2)

        # Set a pose (move platform slightly)
        send_command(sock, {"type": "pose", "x_mm": 10.0, "y_mm": 5.0, "z_mm": -20.0, "roll_deg": 5.0, "pitch_deg": 2.0})
        time.sleep(5)

        # Set another pose
        send_command(sock, {"type": "pose", "x_mm": -10.0, "y_mm": -5.0, "z_mm": -10.0, "roll_deg": -5.0, "pitch_deg": -2.0})
        time.sleep(5)

        # Disable
        send_command(sock, {"type": "state", "value": "disable"})
        time.sleep(1)

        sock.close()
        print("Test completed.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
