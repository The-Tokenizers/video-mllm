import cv2
import numpy as np
from scipy.signal import find_peaks


# ===== Core Frame Extraction Methods =====


def preprocess_frame(frame):
    """Common frame preprocessing"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    return frame


def calculate_action_change(prev_frame, current_frame):
    """Calculate action change score between frames"""
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        current_gray,
        None,  # No flow initialization
        0.5,  # pyramid scale
        3,  # pyramid levels
        15,  # window size
        3,  # iterations
        5,  # neighborhood size
        1.2,  # Gaussian std
        0,  # flags
    )

    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return magnitude.mean()


# ===== Capability-Specific Extraction Methods =====


def uniform_sample_frames(video_path, num_frames):
    """Samples frames uniformly spaced throughout the video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    # Calculate sampling interval
    interval = max(1, total_frames // num_frames)

    for i in range(total_frames):
        ret = cap.grab()
        if i % interval == 0 and ret:
            ret, frame = cap.retrieve()
            if ret:
                frames.append(preprocess_frame(frame))
        if len(frames) >= num_frames:
            break

    cap.release()
    return np.stack(frames[:num_frames])


def extract_action_frames(video_path, num_frames):
    """Extracts frames with peak action changes (for plot/action analysis)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    action_scores = []

    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        frames.append(frame)

        if prev_frame is not None:
            action_scores.append(calculate_action_change(prev_frame, frame))
        prev_frame = frame

    cap.release()

    # Select frames with peak action changes
    peaks, _ = find_peaks(action_scores, distance=len(action_scores) // num_frames)
    selected = [frames[i + 1] for i in peaks if i + 1 < len(frames)]

    # Fill gaps if needed
    if len(selected) < num_frames:
        fill_indices = np.linspace(
            0, len(frames) - 1, num_frames - len(selected), dtype=int
        )
        selected.extend([frames[i] for i in fill_indices])

    return np.stack(selected[:num_frames])


def detect_face_frames(video_path, num_frames):
    """Extracts frames with clearest face visibility"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(video_path)
    frames = []
    face_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            frames.append(frame)
            # Score based on face size and position
            face_scores.append(max([w * h for (x, y, w, h) in faces]))
        else:
            face_scores.append(0)

    cap.release()

    # Select frames with best face visibility
    if len(frames) > num_frames:
        indices = np.argsort(face_scores)[-num_frames:]
        return np.stack([frames[i] for i in indices])
    return np.stack(frames[:num_frames])


def track_motion_frames(video_path, num_frames):
    """Extracts frames with significant motion (for displacement analysis)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    motion_vectors = []

    # First pass: detect motion patterns
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frames.append(frame)

        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            motion_vectors.append(np.mean(np.abs(flow)))
        prev_frame = gray

    cap.release()

    # Select frames with significant motion
    threshold = np.percentile(motion_vectors, 75)
    selected = [
        frames[i + 1]
        for i in range(len(motion_vectors))
        if motion_vectors[i] > threshold
    ][:num_frames]

    # Fill with uniform samples if needed
    if len(selected) < num_frames:
        fill_indices = np.linspace(
            0, len(frames) - 1, num_frames - len(selected), dtype=int
        )
        selected.extend([frames[i] for i in fill_indices])

    return np.stack(selected[:num_frames])


def multi_object_tracking_frames(video_path, num_frames):
    """Extracts frames with multiple detected objects"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    object_counts = []

    # Initialize object detector (simplified example)
    detector = cv2.createBackgroundSubtractorMOG2()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame)
        fg_mask = detector.apply(frame)
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frames.append(frame)
        object_counts.append(len([c for c in contours if cv2.contourArea(c) > 500]))

    cap.release()

    # Select frames with multiple objects
    if len(frames) > num_frames:
        indices = np.argsort(object_counts)[-num_frames:]
        return np.stack([frames[i] for i in indices])
    return np.stack(frames[:num_frames])
