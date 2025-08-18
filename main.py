#!/usr/bin/env python3
import cv2
import time
import threading
import numpy as np
import mediapipe as mp

class VideoStream:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
        self.cap.release()
    def read(self):
        with self.lock:
            return None if not self.grabbed else self.frame.copy()
    def stop(self):
        self.stopped = True

class FaceMeshApp:
    PROCESS_WIDTH = 640
    MAX_FACES = 2
    DETECTION_CONF = 0.6
    TRACKING_CONF = 0.6
    SMOOTH_ALPHA = 0.6
    DRAW_EVERY = 1

    def __init__(self, src=0):
        self.vs = VideoStream(src)
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(static_image_mode=False, max_num_faces=self.MAX_FACES, refine_landmarks=True, min_detection_confidence=self.DETECTION_CONF, min_tracking_confidence=self.TRACKING_CONF)
        self.prev_landmarks = None
        self.fps = 0.0
        self._t = time.time()

    def _smooth(self, new):
        if self.prev_landmarks is None:
            self.prev_landmarks = new
            return new
        sm = self.SMOOTH_ALPHA * new + (1.0 - self.SMOOTH_ALPHA) * self.prev_landmarks
        self.prev_landmarks = sm
        return sm

    def run(self):
        while True:
            frame = self.vs.read()
            if frame is None:
                time.sleep(0.01)
                continue
            h, w = frame.shape[:2]
            scale = self.PROCESS_WIDTH / float(w)
            ph = int(h * scale)
            proc = cv2.resize(frame, (self.PROCESS_WIDTH, ph))
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pts = np.array([[p.x * proc.shape[1], p.y * proc.shape[0]] for p in lm], dtype=np.float32)
                pts = self._smooth(pts)
                sx = w / float(proc.shape[1])
                sy = h / float(proc.shape[0])
                for i, (x, y) in enumerate(pts):
                    if i % self.DRAW_EVERY != 0:
                        continue
                    xx = int(x * sx)
                    yy = int(y * sy)
                    cv2.circle(frame, (xx, yy), 1, (0, 255, 0), -1)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / (now - self._t)) if now != self._t else self.fps
            self._t = now
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow('FaceMesh Fast & Smooth', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.vs.stop()
        cv2.destroyAllWindows()

def atm_security_check(results, proc, frame, warn_distance=0.25):
    """
    Simple shoulder-surfing detector:
    - Triggers when at least two faces are detected.
    - Computes center distance between the first two faces (relative to processing width).
    - If distance < warn_distance -> draws warning on `frame` and returns True.
    Tunables: warn_distance (0.0-1.0) smaller -> more sensitive.
    """
    if not results or not results.multi_face_landmarks or len(results.multi_face_landmarks) < 2:
        return False
    centers = []
    for face in results.multi_face_landmarks[:2]:
        xs = [p.x for p in face.landmark]
        ys = [p.y for p in face.landmark]
        cx = int(np.mean(xs) * proc.shape[1])
        cy = int(np.mean(ys) * proc.shape[0])
        centers.append((cx, cy))
    d = np.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1])
    rel = d / float(proc.shape[1])
    if rel < warn_distance:
        h, w = frame.shape[:2]
        sx = w / float(proc.shape[1])
        sy = h / float(proc.shape[0])
        c1 = (int(centers[0][0] * sx), int(centers[0][1] * sy))
        c2 = (int(centers[1][0] * sx), int(centers[1][1] * sy))
        cv2.line(frame, c1, c2, (0, 0, 255), 2)
        cv2.putText(frame, "WARNING: Possible shoulder-surfing detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return True
    return False

if __name__ == '__main__':
    app = FaceMeshApp(0)
    app.run()
