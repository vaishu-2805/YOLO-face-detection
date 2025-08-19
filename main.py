#!/usr/bin/env python3
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
from datetime import datetime

from ultralytics import YOLO


YOLO_MODEL = 'yolov12n-face.pt'  # Use yolov12n-face.pt as requested
CONFIDENCE_THRESHOLD = 0.3  # Lowered to make detection more sensitive
NMS_THRESHOLD = 0.4
CAMERA_ID = 0
SCREENSHOT_DIR = 'security_screenshots'

# New: Added frame size parameters
PROCESS_WIDTH = 640  # Normal frame size for better analysis
PROCESS_HEIGHT = 480  # Normal frame size for better analysis
MIN_FACE_SIZE = 10  # Lowered to make detection more sensitive
FRAME_SKIP = 4  # Moderate skipping to balance FPS and responsiveness

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def take_screenshot(frame, reason="security_breach"):
    ensure_dir(SCREENSHOT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_DIR}/{reason}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename

class ATMKeypadWindow:
    def __init__(self, root):
        self.root = root
        self.pin_entry = ""
        self.security_breach = False
        self.root.title("ATM Security Prototype - PIN Entry")
        self.root.geometry("400x600")
        self.root.configure(bg='#2c3e50')
        self.create_ui()

    def create_ui(self):
        title_label = tk.Label(self.root, text="🏧 ATM SECURITY PROTOTYPE", font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        self.security_status = tk.Label(self.root, text="🟢 SECURE - Single User Detected", font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#27ae60')
        self.security_status.pack(pady=10)
        self.pin_display = tk.Label(self.root, text="Enter PIN: ____", font=('Arial', 14), bg='#34495e', fg='white', relief='sunken', width=20, height=2)
        self.pin_display.pack(pady=20)
        keypad_frame = tk.Frame(self.root, bg='#2c3e50')
        keypad_frame.pack(pady=20)
        buttons = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9'], ['*', '0', '#']]
        for i, row in enumerate(buttons):
            for j, num in enumerate(row):
                btn = tk.Button(keypad_frame, text=num, font=('Arial', 16, 'bold'), width=5, height=2, bg='#3498db', fg='white', command=lambda n=num: self.keypad_press(n))
                btn.grid(row=i, column=j, padx=5, pady=5)
        action_frame = tk.Frame(self.root, bg='#2c3e50')
        action_frame.pack(pady=20)
        clear_btn = tk.Button(action_frame, text="CLEAR", font=('Arial', 12, 'bold'), width=10, height=2, bg='#e74c3c', fg='white', command=self.clear_pin)
        clear_btn.pack(side=tk.LEFT, padx=10)
        enter_btn = tk.Button(action_frame, text="ENTER", font=('Arial', 12, 'bold'), width=10, height=2, bg='#27ae60', fg='white', command=self.enter_pin)
        enter_btn.pack(side=tk.LEFT, padx=10)
        instructions = tk.Label(self.root, text="This is a PROTOTYPE demonstration.\nSecurity camera monitors for multiple users.\nTransaction is blocked if breach detected.", font=('Arial', 10), bg='#2c3e50', fg='#bdc3c7', justify=tk.CENTER)
        instructions.pack(pady=20)

    def update_security_status(self, status, face_count=0):
        if status == "secure":
            if face_count == 0:
                self.security_status.config(text="⚪ NO USERS DETECTED", fg='#bdc3c7')
            else:
                self.security_status.config(text="🟢 SECURE - Single User Detected", fg='#27ae60')
            self.security_breach = False
        elif status == "breach":
            self.security_status.config(text=f"🔴 SECURITY BREACH - {face_count} People Detected!", fg='#e74c3c')
            self.security_breach = True

    def keypad_press(self, key):
        if self.security_breach:
            self.show_warning()
            return
        if key.isdigit() and len(self.pin_entry) < 4:
            self.pin_entry += key
            self.update_pin_display()

    def clear_pin(self):
        self.pin_entry = ""
        self.update_pin_display()

    def enter_pin(self):
        if self.security_breach:
            self.show_warning()
            return
        if len(self.pin_entry) == 4:
            messagebox.showinfo("Transaction", f"PIN Entered: {'*' * len(self.pin_entry)}\n\nThis is a PROTOTYPE.\nIn real ATM, transaction would proceed.")
            self.clear_pin()
        else:
            messagebox.showwarning("Invalid PIN", "Please enter a 4-digit PIN.")

    def update_pin_display(self):
        display_text = "Enter PIN: " + "*" * len(self.pin_entry) + "_" * (4 - len(self.pin_entry))
        self.pin_display.config(text=display_text)

    def show_warning(self):
        messagebox.showwarning("SECURITY ALERT", "⚠️ MULTIPLE PEOPLE DETECTED! ⚠️\n\nPlease ensure you are alone\nbefore entering your PIN.\n\nTransaction is paused for security.")

# Run inference and get face boxes using Ultralytics YOLO
def detect_faces_yolo(frame, model):
    results = model(frame)
    boxes = []
    h, w = frame.shape[:2]
    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
    return boxes


def security_monitor(keypad_window):
    model = load_yolo_model(YOLO_MODEL)
    cap = cv2.VideoCapture(CAMERA_ID)
    breach_count = 0
    last_breach_time = 0
    analytics = []
    ensure_dir(SCREENSHOT_DIR)
    frame_count = 0
    fps = 0.0
    _t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        scale = PROCESS_WIDTH / float(w)
        ph = PROCESS_HEIGHT
        proc = cv2.resize(frame, (PROCESS_WIDTH, ph))
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        faces = detect_faces_yolo(proc, model)
        face_count = len(faces)
        breach = face_count > 1
        keypad_window.update_security_status("breach" if breach else "secure", face_count)
        analytics.append(face_count)
        # Draw face boxes
        for i, (x, y, w, h) in enumerate(faces):
            color = (0, 0, 255) if breach else (0, 255, 0)
            cv2.rectangle(proc, (x, y), (x + w, y + h), color, 2)
            cv2.putText(proc, f"Person {i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # Breach warning
        if breach:
            cv2.putText(proc, f"SECURITY BREACH - {face_count} PEOPLE DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if time.time() - last_breach_time > 2:
                filename = take_screenshot(proc)
                print(f"Screenshot saved: {filename}")
                breach_count += 1
                last_breach_time = time.time()
        else:
            cv2.putText(proc, f"SECURE - {face_count} person(s) detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(proc, f"Security Breaches: {breach_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if analytics:
            avg_faces = sum(analytics) / len(analytics)
            cv2.putText(proc, f"Avg Faces: {avg_faces:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - _t)) if now != _t else fps
        _t = now
        cv2.putText(proc, f"FPS: {fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('ATM Security Camera', proc)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    return model

def main():
    root = tk.Tk()
    keypad_window = ATMKeypadWindow(root)
    t = threading.Thread(target=security_monitor, args=(keypad_window,), daemon=True)
    t.start()
    root.mainloop()

if __name__ == '__main__':
    main()