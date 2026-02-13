import cv2
import sys
import subprocess
import os
import time
import numpy as np
from collections import OrderedDict
from ultralytics import YOLO

# YOLO confidence threshold
CONF_THRESHOLD = 0.3

# COCO class ID for "sports ball"
SPORTS_BALL_CLASS = 32

# Robot tracking constants
MAX_ROBOTS = 6
PROXIMITY_EXPAND = 1.5          # factor to expand robot bbox for "near" detection
LINK_TIMEOUT_SECONDS = 5.0     # seconds before an unscored ball-robot link expires
ROBOT_BOX_COLOR = (255, 150, 0)  # blue-ish BGR for robot outlines
LINK_LINE_COLOR = (0, 165, 255)  # orange BGR for ball-robot link lines

# Adaptive robot tracking constants
REINIT_INTERVAL = 90             # re-init tracker every N frames to refresh appearance
RECOVERY_SEARCH_EXPAND = 2.0    # expand search area by this factor when recovering
RECOVERY_MATCH_THRESHOLD = 0.5  # template matching confidence for auto-recovery
RECOVERY_COOLDOWN = 10          # only attempt recovery every N frames
ROBOT_YOLO_CLASS = None          # set to a YOLO class ID to enable YOLO-based re-detection
LOST_WARNING_SECONDS = 3.0      # show recalibrate warning after robot lost this long

# Path to VIT tracker model (relative to script directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "vitTracker.onnx")


def _create_robot_tracker():
    """Create the best available OpenCV tracker, with fallback chain:
    TrackerVit (best) > TrackerCSRT > TrackerMIL (always available)."""
    # Try VIT first (Vision Transformer — best for rotation/appearance changes)
    if hasattr(cv2, 'TrackerVit') and os.path.isfile(VIT_MODEL_PATH):
        try:
            params = cv2.TrackerVit_Params()
            params.net = VIT_MODEL_PATH
            return cv2.TrackerVit.create(params)
        except cv2.error:
            pass
    # Try CSRT (needs opencv-contrib-python)
    if hasattr(cv2, 'TrackerCSRT'):
        return cv2.TrackerCSRT.create()
    # Fallback to MIL (always available)
    return cv2.TrackerMIL.create()


def _create_kalman(cx, cy, pn_h=1, pn_v=100, meas_n=25):
    """Create a Kalman filter for tracking position, velocity, and acceleration
    (x, y, vx, vy, ax, ay). Tuned for FRC 2026 fuel at ~20 ft/s."""
    dt = 1 / 30  # 30 FPS camera
    kf = cv2.KalmanFilter(6, 2)
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0,  0.5*dt**2, 0        ],
        [0, 1, 0,  dt, 0,         0.5*dt**2 ],
        [0, 0, 1,  0,  dt,        0         ],
        [0, 0, 0,  1,  0,         dt        ],
        [0, 0, 0,  0,  1,         0         ],
        [0, 0, 0,  0,  0,         1         ],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=np.float32)
    kf.processNoiseCov = np.diag(
        [pn_h, pn_v, pn_h, pn_v, pn_h, pn_v]
    ).astype(np.float32)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_n
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 100
    kf.statePost = np.array([[cx], [cy], [0], [0], [0], [0]], dtype=np.float32)
    return kf


class KalmanMOTracker:
    """Multi-object tracker using per-object Kalman filters.
    Uses motion-gated data association: detections are matched to tracks
    based on distance to the Kalman-predicted position, not last-seen position."""

    def __init__(self, max_disappeared=30, gate_distance=200,
                 pn_h=1, pn_v=100, meas_n=25,
                 use_time=False, disappear_seconds=5.0):
        self.next_id = 0
        self.tracks = OrderedDict()
        self.gate_distance = gate_distance
        self.max_disappeared = max_disappeared
        self.pn_h = pn_h
        self.pn_v = pn_v
        self.meas_n = meas_n
        self.use_time = use_time
        self.disappear_seconds = disappear_seconds

    def update_noise(self, pn_h, pn_v, meas_n):
        """Live-update Kalman noise matrices on all active tracks."""
        self.pn_h = pn_h
        self.pn_v = pn_v
        self.meas_n = meas_n
        for trk in self.tracks.values():
            trk["kf"].processNoiseCov = np.diag(
                [pn_h, pn_v, pn_h, pn_v, pn_h, pn_v]
            ).astype(np.float32)
            trk["kf"].measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_n

    def _register(self, cx, cy, bbox):
        self.tracks[self.next_id] = {
            "kf": _create_kalman(float(cx), float(cy),
                                 self.pn_h, self.pn_v, self.meas_n),
            "bbox": bbox,
            "disappeared": 0,
            "disappeared_since": None,
            "predicted": (float(cx), float(cy)),
        }
        self.next_id += 1

    def _deregister(self, oid):
        del self.tracks[oid]

    def _should_deregister(self, oid, current_time):
        """Check if a track should be deregistered based on mode."""
        if self.use_time and current_time is not None:
            since = self.tracks[oid]["disappeared_since"]
            return since is not None and (current_time - since) > self.disappear_seconds
        return self.tracks[oid]["disappeared"] > self.max_disappeared

    def update(self, detections, current_time=None):
        """Update with new detections: list of (cx, cy, x, y, w, h).
        Returns dict of id -> (cx, cy, px, py, x, y, w, h).
        current_time: pass time.time() for camera mode (time-based disappearance)."""

        for oid, trk in self.tracks.items():
            pred = trk["kf"].predict()
            trk["predicted"] = (float(pred[0, 0]), float(pred[1, 0]))

        if len(detections) == 0:
            for oid in list(self.tracks.keys()):
                self.tracks[oid]["disappeared"] += 1
                if self.tracks[oid]["disappeared_since"] is None:
                    self.tracks[oid]["disappeared_since"] = current_time
                if self._should_deregister(oid, current_time):
                    self._deregister(oid)
            return self._build_result()

        input_centroids = np.array([(d[0], d[1]) for d in detections])
        input_bboxes = [(d[2], d[3], d[4], d[5]) for d in detections]

        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self._register(input_centroids[i][0], input_centroids[i][1],
                               input_bboxes[i])
            return self._build_result()

        track_ids = list(self.tracks.keys())
        predicted_positions = np.array([self.tracks[oid]["predicted"]
                                        for oid in track_ids])

        diff = predicted_positions[:, np.newaxis] - input_centroids[np.newaxis, :]
        D = np.sqrt((diff ** 2).sum(axis=2))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.gate_distance:
                continue
            oid = track_ids[row]
            measurement = np.array([[input_centroids[col][0]],
                                     [input_centroids[col][1]]], dtype=np.float32)
            self.tracks[oid]["kf"].correct(measurement)
            self.tracks[oid]["bbox"] = input_bboxes[col]
            self.tracks[oid]["disappeared"] = 0
            self.tracks[oid]["disappeared_since"] = None
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(track_ids))) - used_rows:
            oid = track_ids[row]
            self.tracks[oid]["disappeared"] += 1
            if self.tracks[oid]["disappeared_since"] is None:
                self.tracks[oid]["disappeared_since"] = current_time
            if self._should_deregister(oid, current_time):
                self._deregister(oid)

        for col in set(range(len(input_centroids))) - used_cols:
            self._register(input_centroids[col][0], input_centroids[col][1],
                           input_bboxes[col])

        return self._build_result()

    def _build_result(self):
        result = {}
        for oid, trk in self.tracks.items():
            state = trk["kf"].statePost
            cx, cy = float(state[0, 0]), float(state[1, 0])
            px, py = trk["predicted"]
            x, y, w, h = trk["bbox"]
            result[oid] = (cx, cy, px, py, x, y, w, h)
        return result


MIN_CONTOUR_AREA = 10


def extract_hsv_range(roi_bgr):
    """Extract an adaptive HSV color range from a BGR ROI."""
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_roi)

    h_med, s_med, v_med = np.median(h), np.median(s), np.median(v)
    h_std, s_std, v_std = np.std(h), np.std(s), np.std(v)

    h_margin = max(h_std * 2, 10)
    s_margin = max(s_std * 2, 40)
    v_margin = max(v_std * 2, 40)

    lower = np.array([
        max(0, h_med - h_margin),
        max(50, s_med - s_margin),
        max(50, v_med - v_margin),
    ], dtype=np.uint8)

    upper = np.array([
        min(179, h_med + h_margin),
        255,
        255,
    ], dtype=np.uint8)

    return lower, upper


def detect_hsv(frame, hsv_lower, hsv_upper, min_area=MIN_CONTOUR_AREA,
               kernel_size=9):
    """Detect all objects matching an HSV color range.
    Returns list of (cx, cy, x, y, w, h)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if hsv_lower[0] > hsv_upper[0]:
        mask1 = cv2.inRange(hsv, np.array([0, hsv_lower[1], hsv_lower[2]]),
                            np.array([hsv_upper[0], 255, 255]))
        mask2 = cv2.inRange(hsv, hsv_lower,
                            np.array([179, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    ks = max(1, kernel_size | 1)  # force odd, minimum 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area or cv2.contourArea(contour) > 900:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        detections.append((cx, cy, x, y, w, h))

    return detections


def detect_yolo(frame, model, conf=CONF_THRESHOLD):
    """Detect all sports balls using YOLO.
    Returns list of (cx, cy, x, y, w, h)."""
    results = model(frame, classes=[SPORTS_BALL_CLASS], conf=conf, verbose=False)

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w // 2
        cy = y1 + h // 2
        detections.append((cx, cy, x1, y1, w, h))

    return detections


def _try_recover_robot(frame, robot, model=None):
    """Try to recover a lost robot using template matching, then optionally YOLO.
    Updates robot['bbox'] in-place and returns True if recovered."""
    last_crop = robot.get("last_good_crop")
    last_bbox = robot.get("last_known_bbox")
    if last_crop is None or last_bbox is None:
        return False

    rx, ry, rw, rh = last_bbox
    fh, fw = frame.shape[:2]

    # Define expanded search region around last known position
    expand_w = int(rw * RECOVERY_SEARCH_EXPAND)
    expand_h = int(rh * RECOVERY_SEARCH_EXPAND)
    sx = max(0, rx - expand_w)
    sy = max(0, ry - expand_h)
    sw = min(fw - sx, rw + 2 * expand_w)
    sh = min(fh - sy, rh + 2 * expand_h)

    search_region = frame[sy:sy + sh, sx:sx + sw]
    template = last_crop

    # Template must be smaller than the search region
    if (template.shape[0] < search_region.shape[0] and
            template.shape[1] < search_region.shape[1] and
            template.shape[0] > 0 and template.shape[1] > 0):
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= RECOVERY_MATCH_THRESHOLD:
            mx, my = max_loc
            new_bbox = (sx + mx, sy + my, template.shape[1], template.shape[0])
            robot["bbox"] = new_bbox
            robot["last_known_bbox"] = new_bbox
            return True

    # Optionally try YOLO-based re-detection
    if model is not None and ROBOT_YOLO_CLASS is not None:
        results = model(frame, classes=[ROBOT_YOLO_CLASS], conf=0.3, verbose=False)
        best_dist = float('inf')
        best_bbox = None
        last_cx = rx + rw // 2
        last_cy = ry + rh // 2
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            det_cx = (x1 + x2) // 2
            det_cy = (y1 + y2) // 2
            dist = ((det_cx - last_cx) ** 2 + (det_cy - last_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_bbox = (x1, y1, x2 - x1, y2 - y1)
        if best_bbox is not None and best_dist < max(rw, rh) * RECOVERY_SEARCH_EXPAND:
            robot["bbox"] = best_bbox
            robot["last_known_bbox"] = best_bbox
            return True

    return False


def select_video_source():
    """Prompt the user to choose a video source.
    Returns (cap, is_camera) where is_camera is True for webcam."""
    print("\n=== Select Video Source ===")
    print("1. Webcam (default camera)")
    print("2. Local video file")
    print("3. YouTube video")
    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        return cv2.VideoCapture(0), True
    elif choice == "2":
        path = input("Enter path to video file: ").strip().strip('"')
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{path}'.")
            sys.exit(1)
        return cap, False
    elif choice == "3":
        url = input("Enter YouTube URL: ").strip()
        print("Downloading video with yt-dlp, please wait...")
        videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Videos")
        os.makedirs(videos_dir, exist_ok=True)
        output_template = os.path.join(videos_dir, "%(title)s.%(ext)s")
        try:
            subprocess.run(
                [sys.executable, "-m", "yt_dlp",
                 "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                 "-o", output_template, url],
                check=True,
            )
            result = subprocess.run(
                [sys.executable, "-m", "yt_dlp",
                 "--print", "filename",
                 "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                 "-o", output_template, url],
                capture_output=True, text=True, check=True,
            )
            output_path = result.stdout.strip()
        except subprocess.CalledProcessError:
            print("Error: Failed to download video. Check the URL and try again.")
            sys.exit(1)
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            print("Error: Could not open downloaded video.")
            sys.exit(1)
        print("Download complete. Starting detector...")
        return cap, False
    else:
        print("Invalid choice. Using webcam.")
        return cv2.VideoCapture(0), True


def main():
    # Load YOLO model (auto-downloads on first run)
    print("Loading YOLO model...")
    model = YOLO("yolo26n.pt")
    print("Model loaded.")

    cap, is_camera = select_video_source()

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from video source.")
        sys.exit(1)

    # Create a resizable window
    cv2.namedWindow("YOLO Ball Detector", cv2.WINDOW_NORMAL)

    # Step 1: Select the RED detection zone
    print("\nStep 1: Select the RED detection zone, then press ENTER or SPACE.")
    red_zone = cv2.selectROI("YOLO Ball Detector", frame, fromCenter=False, showCrosshair=True)
    if red_zone == (0, 0, 0, 0):
        print("No RED zone selected. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    # Step 2: Select the BLUE detection zone
    print("Step 2: Select the BLUE detection zone, then press ENTER or SPACE.")
    blue_zone = cv2.selectROI("YOLO Ball Detector", frame, fromCenter=False, showCrosshair=True)
    if blue_zone == (0, 0, 0, 0):
        print("No BLUE zone selected. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    # Step 3: Select robot ROIs (up to 6)
    print(f"\nStep 3: Select robot ROIs (up to {MAX_ROBOTS}). "
          "Press ESC or cancel empty selection to stop.")
    robots = {}
    for i in range(MAX_ROBOTS):
        print(f"  Select robot {i+1}/{MAX_ROBOTS} bounding box, "
              "or press ESC to finish.")
        robot_roi = cv2.selectROI("YOLO Ball Detector", frame,
                                   fromCenter=False, showCrosshair=True)
        if robot_roi == (0, 0, 0, 0):
            print(f"  No more robots. {len(robots)} robots registered.")
            break
        name = input(f"  Enter team number/name for robot {i+1}: ").strip()
        if not name:
            name = f"Robot{i+1}"
        tracker = _create_robot_tracker()
        roi_tuple = tuple(int(v) for v in robot_roi)
        tracker.init(frame, roi_tuple)
        rx, ry, rw, rh = roi_tuple
        fh, fw = frame.shape[:2]
        cx1, cy1 = max(0, rx), max(0, ry)
        cx2, cy2 = min(fw, rx + rw), min(fh, ry + rh)
        initial_crop = frame[cy1:cy2, cx1:cx2].copy() if cx2 > cx1 and cy2 > cy1 else None
        robots[i] = {
            "name": name,
            "tracker": tracker,
            "bbox": roi_tuple,
            "initial_size": (roi_tuple[2], roi_tuple[3]),
            "ok": True,
            "red_score": 0,
            "blue_score": 0,
            "last_good_crop": initial_crop,
            "last_known_bbox": roi_tuple,
            "frames_since_reinit": 0,
            "recovery_counter": 0,
            "lost_since": None,
            "lost_warned": False,
        }
        print(f"  Registered: {name}")

    ball_links = {}       # ball_id -> {robot_idx, link_time}
    ball_near_robot = {}  # ball_id -> robot_idx (current frame proximity)

    # --- Tuning Controls Window ---
    CTRL_WIN = "Controls"
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    noop = lambda x: None
    # Detection
    cv2.createTrackbar("YOLO Conf %",  CTRL_WIN, 30,   95,   noop)
    cv2.createTrackbar("Min Area",     CTRL_WIN, 50,   500, noop)
    cv2.createTrackbar("Kernel Size",  CTRL_WIN, 3,    31,   noop)
    # Tracking
    cv2.createTrackbar("Gate Dist",    CTRL_WIN, 30,  5000,  noop)
    cv2.createTrackbar("Max Disappear",CTRL_WIN, 30,   500,  noop)
    # Kalman noise
    cv2.createTrackbar("PN Horiz",     CTRL_WIN, 1,    200,  noop)
    cv2.createTrackbar("PN Vert",      CTRL_WIN, 100,  200,  noop)
    cv2.createTrackbar("Meas Noise",   CTRL_WIN, 25,   200,  noop)
    # Playback
    cv2.createTrackbar("Speed Factor", CTRL_WIN, 1250, 5000, noop)

    # Video FPS for playback speed and disappearance calculation
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 30

    # Kalman MOT tracker for persistent object IDs with motion prediction
    # Camera mode: use wall-clock time (5 seconds real time)
    # Video mode: use frame count (5 seconds of video time = 5 * fps frames)
    if is_camera:
        disappear_frames = int(5 * 30)  # fallback frame count
        ct = KalmanMOTracker(max_disappeared=disappear_frames,
                             use_time=True, disappear_seconds=5.0)
    else:
        disappear_frames = int(5 * video_fps)
        ct = KalmanMOTracker(max_disappeared=disappear_frames)
    cv2.setTrackbarPos("Max Disappear", CTRL_WIN, disappear_frames)

    # Detection modes
    yolo_enabled = True
    color_targets = []  # list of (hsv_lower, hsv_upper)

    # Per-object zone entry state
    red_count = 0
    blue_count = 0
    was_inside_red = {}
    was_inside_blue = {}

    print("\n=== Keyboard Controls ===")
    print("q - Quit")
    print("a - Add a color target (select object ROI)")
    print("b - Undo last color target")
    print("y - Toggle YOLO detection on/off")
    print("r - Reset tracker, color targets, and robot scores")
    print("1 - Re-draw RED zone")
    print("2 - Re-draw BLUE zone")
    print("3 - Re-initialize robot trackers")
    print("=========================")
    print("Use the Controls window sliders to tune parameters in real time.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # --- Read trackbar values ---
        conf = max(5, cv2.getTrackbarPos("YOLO Conf %", CTRL_WIN)) / 100.0
        min_area = cv2.getTrackbarPos("Min Area", CTRL_WIN)
        kernel_sz = cv2.getTrackbarPos("Kernel Size", CTRL_WIN)
        gate_dist = max(50, cv2.getTrackbarPos("Gate Dist", CTRL_WIN))
        max_disap = max(1, cv2.getTrackbarPos("Max Disappear", CTRL_WIN))
        pn_h = max(1, cv2.getTrackbarPos("PN Horiz", CTRL_WIN))
        pn_v = max(1, cv2.getTrackbarPos("PN Vert", CTRL_WIN))
        meas_n = max(1, cv2.getTrackbarPos("Meas Noise", CTRL_WIN))
        speed_factor = max(1, cv2.getTrackbarPos("Speed Factor", CTRL_WIN))
        frame_delay = max(1, int(speed_factor / video_fps))

        # Apply to tracker
        ct.gate_distance = gate_dist
        ct.max_disappeared = max_disap
        ct.update_noise(pn_h, pn_v, meas_n)

        # --- Update robot trackers ---
        fh_r, fw_r = frame.shape[:2]
        for idx, robot in robots.items():
            ok, new_bbox = robot["tracker"].update(frame)
            if ok:
                # Lock bbox to initial size — keep the tracker's center,
                # but force width/height to the original selection
                tx, ty, tw, th = (int(v) for v in new_bbox)
                iw, ih = robot["initial_size"]
                center_x = tx + tw // 2
                center_y = ty + th // 2
                locked_x = center_x - iw // 2
                locked_y = center_y - ih // 2

                # Reject if center is outside the frame (robot left view)
                if not (0 <= center_x < fw_r and 0 <= center_y < fh_r):
                    ok = False

            if ok:
                robot["ok"] = True
                robot["bbox"] = (locked_x, locked_y, iw, ih)
                robot["frames_since_reinit"] += 1
                robot["recovery_counter"] = 0
                robot["lost_since"] = None
                robot["lost_warned"] = False
                # Save last good crop for recovery (clipped to frame)
                cx1, cy1 = max(0, locked_x), max(0, locked_y)
                cx2, cy2 = min(fw_r, locked_x + iw), min(fh_r, locked_y + ih)
                if cx2 > cx1 and cy2 > cy1:
                    robot["last_good_crop"] = frame[cy1:cy2, cx1:cx2].copy()
                    robot["last_known_bbox"] = robot["bbox"]
                # Periodic re-init to refresh appearance model
                if robot["frames_since_reinit"] >= REINIT_INTERVAL:
                    robot["tracker"] = _create_robot_tracker()
                    robot["tracker"].init(frame, robot["bbox"])
                    robot["frames_since_reinit"] = 0
            else:
                # Mark lost and track how long
                robot["ok"] = False
                if robot["lost_since"] is None:
                    robot["lost_since"] = time.time()
                # Console warning once per loss event
                if not robot["lost_warned"] and robot["lost_since"] is not None:
                    elapsed = time.time() - robot["lost_since"]
                    if elapsed >= LOST_WARNING_SECONDS:
                        print(f"  WARNING: {robot['name']} lost for "
                              f"{elapsed:.1f}s — press 3 to recalibrate")
                        robot["lost_warned"] = True
                # Attempt auto-recovery
                robot["recovery_counter"] += 1
                if robot["recovery_counter"] % RECOVERY_COOLDOWN == 1:
                    yolo_model = model if ROBOT_YOLO_CLASS is not None else None
                    if _try_recover_robot(frame, robot, yolo_model):
                        robot["tracker"] = _create_robot_tracker()
                        robot["tracker"].init(frame, robot["bbox"])
                        robot["ok"] = True
                        robot["frames_since_reinit"] = 0
                        robot["recovery_counter"] = 0
                        robot["lost_since"] = None
                        robot["lost_warned"] = False

        # Detect objects from all active sources
        all_detections = []
        if yolo_enabled:
            all_detections.extend(detect_yolo(frame, model, conf))
        for hsv_lower, hsv_upper in color_targets:
            all_detections.extend(detect_hsv(frame, hsv_lower, hsv_upper,
                                             min_area, kernel_sz))
        tracked = ct.update(all_detections, current_time=time.time())

        # Zone coordinates
        rz_x, rz_y, rz_w, rz_h = [int(v) for v in red_zone]
        bz_x, bz_y, bz_w, bz_h = [int(v) for v in blue_zone]

        # Clean up stale IDs from zone state
        active_ids = set(tracked.keys())
        for stale_id in list(was_inside_red.keys()):
            if stale_id not in active_ids:
                del was_inside_red[stale_id]
                del was_inside_blue[stale_id]
        for stale_id in list(ball_links.keys()):
            if stale_id not in active_ids:
                del ball_links[stale_id]
        for stale_id in list(ball_near_robot.keys()):
            if stale_id not in active_ids:
                del ball_near_robot[stale_id]

        # --- Ball-robot proximity and link management ---
        now = time.time()

        # Phase 1: Which balls are currently near which robots?
        current_near = {}
        for ball_id, (cx, cy, px, py, ox, oy, ow, oh) in tracked.items():
            for idx, robot in robots.items():
                if not robot["ok"]:
                    continue
                rx, ry, rw, rh = robot["bbox"]
                expand_w = rw * (PROXIMITY_EXPAND - 1) / 2
                expand_h = rh * (PROXIMITY_EXPAND - 1) / 2
                ex, ey = rx - expand_w, ry - expand_h
                ew, eh = rw + 2 * expand_w, rh + 2 * expand_h
                if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
                    current_near[ball_id] = idx
                    break

        # Phase 2: Detect departures (ball was near robot, now isn't)
        for ball_id, prev_robot_idx in list(ball_near_robot.items()):
            if ball_id not in tracked:
                continue
            if ball_id not in current_near and ball_id not in ball_links:
                ball_links[ball_id] = {
                    "robot_idx": prev_robot_idx,
                    "link_time": now,
                }

        # Phase 3: Resolve links (score or timeout)
        for ball_id in list(ball_links.keys()):
            if ball_id not in tracked:
                del ball_links[ball_id]
                continue
            link = ball_links[ball_id]
            bcx, bcy = tracked[ball_id][0], tracked[ball_id][1]
            # Check RED zone
            if (rz_x <= bcx <= rz_x + rz_w) and (rz_y <= bcy <= rz_y + rz_h):
                robots[link["robot_idx"]]["red_score"] += 1
                del ball_links[ball_id]
                continue
            # Check BLUE zone
            if (bz_x <= bcx <= bz_x + bz_w) and (bz_y <= bcy <= bz_y + bz_h):
                robots[link["robot_idx"]]["blue_score"] += 1
                del ball_links[ball_id]
                continue
            # Timeout
            if now - link["link_time"] > LINK_TIMEOUT_SECONDS:
                del ball_links[ball_id]

        ball_near_robot = current_near

        # Zone highlight flags
        any_in_red = False
        any_in_blue = False

        # Process each tracked object
        for oid, (cx, cy, px, py, ox, oy, ow, oh) in tracked.items():
            # Draw bounding box with ID
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {oid}", (ox, oy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw predicted next position (small yellow crosshair)
            ipx, ipy = int(px), int(py)
            cv2.drawMarker(frame, (ipx, ipy), (0, 255, 255),
                           cv2.MARKER_CROSS, 12, 2)

            # Draw motion trail: line from current to predicted
            cv2.line(frame, (int(cx), int(cy)), (ipx, ipy), (0, 255, 255), 1)

            # Draw ball-robot link line
            if oid in ball_links:
                link = ball_links[oid]
                robot = robots[link["robot_idx"]]
                if robot["ok"]:
                    r_rx, r_ry, r_rw, r_rh = robot["bbox"]
                    rcx = int(r_rx + r_rw / 2)
                    rcy = int(r_ry + r_rh / 2)
                    cv2.line(frame, (int(cx), int(cy)), (rcx, rcy),
                             LINK_LINE_COLOR, 2)
                    mid_x = int((cx + rcx) / 2)
                    mid_y = int((cy + rcy) / 2)
                    cv2.putText(frame, robot["name"], (mid_x, mid_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                LINK_LINE_COLOR, 1)

            # RED zone check
            in_red = (rz_x <= cx <= rz_x + rz_w) and (rz_y <= cy <= rz_y + rz_h)
            if in_red and not was_inside_red.get(oid, False):
                red_count += 1
            was_inside_red[oid] = in_red
            if in_red:
                any_in_red = True

            # BLUE zone check
            in_blue = (bz_x <= cx <= bz_x + bz_w) and (bz_y <= cy <= bz_y + bz_h)
            if in_blue and not was_inside_blue.get(oid, False):
                blue_count += 1
            was_inside_blue[oid] = in_blue
            if in_blue:
                any_in_blue = True

        # Draw zones with highlight
        red_draw = (0, 0, 255) if any_in_red else (0, 0, 200)
        blue_draw = (255, 0, 0) if any_in_blue else (200, 0, 0)

        cv2.rectangle(frame, (rz_x, rz_y), (rz_x + rz_w, rz_y + rz_h), red_draw, 2)
        cv2.putText(frame, "RED", (rz_x, rz_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, red_draw, 2)

        cv2.rectangle(frame, (bz_x, bz_y), (bz_x + bz_w, bz_y + bz_h), blue_draw, 2)
        cv2.putText(frame, "BLUE", (bz_x, bz_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, blue_draw, 2)

        # HUD
        cv2.putText(frame, f"RED Zone Count: {red_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"BLUE Zone Count: {blue_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Objects: {len(tracked)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show active detection modes
        yolo_status = f"YOLO: ON ({conf:.2f})" if yolo_enabled else "YOLO: OFF"
        yolo_color = (0, 255, 0) if yolo_enabled else (0, 0, 200)
        cv2.putText(frame, yolo_status, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, yolo_color, 2)
        hsv_status = f"Color targets: {len(color_targets)}"
        hsv_color = (0, 255, 0) if color_targets else (150, 150, 150)
        cv2.putText(frame, hsv_status, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hsv_color, 2)
        display_fps = video_fps or 30
        cv2.putText(frame, f"FPS: {display_fps:.0f}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame,
                    f"Gate:{gate_dist} PN:{pn_h}/{pn_v} MN:{meas_n}",
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # --- Draw robots ---
        for idx, robot in robots.items():
            if not robot["ok"]:
                # Draw last known position with dashed-style outline
                if robot["last_known_bbox"] is not None:
                    rx, ry, rw, rh = robot["last_known_bbox"]
                    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                                  (0, 0, 255), 1)
                    cv2.putText(frame, f"{robot['name']} LOST",
                                (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)
                continue
            rx, ry, rw, rh = robot["bbox"]
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                          ROBOT_BOX_COLOR, 2)
            cv2.putText(frame, robot["name"], (rx, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROBOT_BOX_COLOR, 2)

        # --- Recalibration warning banner ---
        lost_names = [r["name"] for r in robots.values()
                      if not r["ok"] and r["lost_since"] is not None
                      and (time.time() - r["lost_since"]) >= LOST_WARNING_SECONDS]
        if lost_names:
            # Flash by toggling visibility every ~0.5s
            flash_on = int(time.time() * 2) % 2 == 0
            if flash_on:
                warn_text = f"RECALIBRATE (3): {', '.join(lost_names)}"
                tw = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.8, 2)[0][0]
                banner_x = (frame.shape[1] - tw) // 2
                banner_y = frame.shape[0] - 30
                # Dark background for readability
                cv2.rectangle(frame, (banner_x - 10, banner_y - 30),
                              (banner_x + tw + 10, banner_y + 10),
                              (0, 0, 0), -1)
                cv2.putText(frame, warn_text, (banner_x, banner_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Per-robot scoring HUD (right side) ---
        if robots:
            frame_w = frame.shape[1]
            hud_x = frame_w - 300
            cv2.putText(frame, "ROBOT SCORES", (hud_x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            for i, (idx, robot) in enumerate(robots.items()):
                y_pos = 60 + i * 30
                score_text = f"{robot['name']}: R={robot['red_score']} B={robot['blue_score']}"
                color = ROBOT_BOX_COLOR
                if not robot["ok"]:
                    score_text += " [LOST]"
                    color = (100, 100, 100)
                cv2.putText(frame, score_text, (hud_x, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("YOLO Ball Detector", frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("a"):
            # Add a color target by selecting an object ROI
            new_bbox = cv2.selectROI("YOLO Ball Detector", frame,
                                     fromCenter=False, showCrosshair=True)
            if new_bbox != (0, 0, 0, 0):
                nx, ny, nw, nh = [int(v) for v in new_bbox]
                roi = frame[ny:ny+nh, nx:nx+nw]
                color_targets.append(extract_hsv_range(roi))
                print(f"Added color target #{len(color_targets)}")
        elif key == ord("b"):
            # Undo last color target
            if color_targets:
                color_targets.pop()
                ct = KalmanMOTracker(max_disappeared=disappear_frames,
                                     use_time=is_camera, disappear_seconds=5.0)
                was_inside_red.clear()
                was_inside_blue.clear()
                print(f"Removed last color target. {len(color_targets)} remaining.")
            else:
                print("No color targets to remove.")
        elif key == ord("y"):
            # Toggle YOLO on/off
            yolo_enabled = not yolo_enabled
            ct = KalmanMOTracker(max_disappeared=disappear_frames,
                                 use_time=is_camera, disappear_seconds=5.0)
            was_inside_red.clear()
            was_inside_blue.clear()
            print(f"YOLO {'enabled' if yolo_enabled else 'disabled'}")
        elif key == ord("r"):
            # Reset tracker, color targets, and robot scores
            ct = KalmanMOTracker(max_disappeared=disappear_frames,
                                 use_time=is_camera, disappear_seconds=5.0)
            color_targets.clear()
            was_inside_red.clear()
            was_inside_blue.clear()
            ball_links.clear()
            ball_near_robot.clear()
            red_count = 0
            blue_count = 0
            for robot in robots.values():
                robot["red_score"] = 0
                robot["blue_score"] = 0
                robot["lost_since"] = None
                robot["lost_warned"] = False
            print("Tracker, color targets, and robot scores reset.")
        elif key == ord("1"):
            # Re-draw RED zone
            new_zone = cv2.selectROI("YOLO Ball Detector", frame,
                                     fromCenter=False, showCrosshair=True)
            if new_zone != (0, 0, 0, 0):
                red_zone = new_zone
                was_inside_red.clear()
        elif key == ord("2"):
            # Re-draw BLUE zone
            new_zone = cv2.selectROI("YOLO Ball Detector", frame,
                                     fromCenter=False, showCrosshair=True)
            if new_zone != (0, 0, 0, 0):
                blue_zone = new_zone
                was_inside_blue.clear()
        elif key == ord("3"):
            # Re-initialize robot trackers
            for idx, robot in robots.items():
                print(f"  Re-select ROI for {robot['name']}:")
                new_roi = cv2.selectROI("YOLO Ball Detector", frame,
                                         fromCenter=False,
                                         showCrosshair=True)
                if new_roi != (0, 0, 0, 0):
                    roi_tuple = tuple(int(v) for v in new_roi)
                    robot["tracker"] = _create_robot_tracker()
                    robot["tracker"].init(frame, roi_tuple)
                    robot["bbox"] = roi_tuple
                    robot["initial_size"] = (roi_tuple[2], roi_tuple[3])
                    robot["ok"] = True
                    robot["frames_since_reinit"] = 0
                    robot["recovery_counter"] = 0
                    robot["lost_since"] = None
                    robot["lost_warned"] = False
                    robot["last_known_bbox"] = roi_tuple
                    rx, ry, rw, rh = roi_tuple
                    fh_r, fw_r = frame.shape[:2]
                    cx1, cy1 = max(0, rx), max(0, ry)
                    cx2, cy2 = min(fw_r, rx + rw), min(fh_r, ry + rh)
                    if cx2 > cx1 and cy2 > cy1:
                        robot["last_good_crop"] = frame[cy1:cy2, cx1:cx2].copy()
                    print(f"  Re-initialized {robot['name']}")
            ball_links.clear()
            ball_near_robot.clear()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
