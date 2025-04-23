import os

import cv2
import numpy as np
import pandas as pd
import torch
from scenedetect import open_video, SceneManager
from scenedetect.detectors import (
    ContentDetector,
    ThresholdDetector,
    AdaptiveDetector,
    HistogramDetector,
    HashDetector,
)
from transformers import (
    AutoConfig,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)

from capability_config import CAPABILITY_CONFIG

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(base_dir, "models", "llava_next_video_7b_dpo_model")
processor_dir = os.path.join(model_dir, "processor")
model_subdir = os.path.join(model_dir, "model")
video_dir = os.path.join(base_dir, "data", "Benchmark-AllVideos-HQ-Encoded-challenge")

processor = LlavaNextVideoProcessor.from_pretrained(processor_dir, use_fast=True)
# config = AutoConfig.from_pretrained(model_subdir)
# config.text_config.rope_scaling = {
#     "factor": 4.0,
#     "rope_type": "linear",
#     "type": "linear"
# }
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_subdir, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()


def detect_scenes(video_path, detector_type, detector_params, downscale_factor=1):
    """Detect scenes using specified detector and parameters"""
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()

        # Initialize the appropriate detector
        detector = {
            "ContentDetector": ContentDetector,
            "ThresholdDetector": ThresholdDetector,
            "AdaptiveDetector": AdaptiveDetector,
            "HistogramDetector": HistogramDetector,
            "HashDetector": HashDetector,
        }.get(detector_type)

        if not detector:
            raise ValueError(f"Unsupported detector type: {detector_type}")

        detector_params = detector_params.copy()

        if "weights" in detector_params and detector_type in [
            "ContentDetector",
            "AdaptiveDetector",
        ]:
            weights_dict = detector_params.pop("weights")
            components = ContentDetector.Components(
                delta_hue=weights_dict.get("delta_hue", 1.0),
                delta_sat=weights_dict.get("delta_sat", 1.0),
                delta_lum=weights_dict.get("delta_lum", 1.0),
                delta_edges=weights_dict.get("delta_edges", 0.0),
            )
            detector_params["weights"] = components

        if "method" in detector_params and detector_type == "ThresholdDetector":
            method_str = detector_params.pop("method")
            method_map = {
                "FLOOR": ThresholdDetector.Method.FLOOR,
                "CEILING": ThresholdDetector.Method.CEILING,
            }
            method_enum = method_map.get(method_str, ThresholdDetector.Method.FLOOR)
            detector_params["method"] = method_enum

        try:
            detector_instance = detector(**detector_params)
        except TypeError as e:
            print(f"Invalid parameters for {detector_type}: {str(e)}")
            return []

        scene_manager.add_detector(detector_instance)

        if downscale_factor > 1:
            scene_manager.downscale = downscale_factor

        scene_manager.detect_scenes(video=video)

        return scene_manager.get_scene_list(start_in_scene=True)

    except Exception as e:
        print(f"Scene detection failed ({detector_type}): {str(e)}")
        return []
    finally:
        pass


def sample_frames_from_video(video_path, scene_list, sampling_method):
    """Optimized frame sampling with modern video handling"""
    try:
        video = open_video(video_path)
        total_frames = video.duration.get_frames()
        frames = []

        # If no scenes detected, fallback to uniform sampling
        if not scene_list or not isinstance(scene_list, list):
            return uniform_sample_frames(video, min(32, total_frames))

        # Process each scene until we have enough frames
        for scene in scene_list:
            if len(frames) >= 32:
                break

            # Skip invalid scenes
            if not isinstance(scene, (tuple, list)) or len(scene) < 2:
                continue

            print(f"Processing scene: {scene}")
            start = int(scene[0].get_frames())
            end = int(scene[1].get_frames())
            scene_len = end - start

            # Determine positions based on sampling method
            if sampling_method == "uniform":
                positions = np.linspace(start, end, num=4, dtype=int)[1:-1]
            elif sampling_method == "keyframe":
                positions = [start, (start + end) // 2]
            elif sampling_method == "dense_keyframes":
                positions = np.linspace(start, end, num=5, dtype=int)[:-1]
            elif sampling_method == "face_detection":
                positions = [
                    start + scene_len // 3,
                    start + scene_len // 2,
                    start + 2 * scene_len // 3,
                ]
            elif sampling_method == "keyframe+uniform":
                positions = list(
                    {
                        start,
                        (start + end) // 2,
                        *np.linspace(start, end, num=4, dtype=int)[1:-1],
                    }
                )
            elif sampling_method == "keyframe+tracking":
                positions = list(
                    {
                        start,
                        (start + end) // 2,
                        *range(start, end, 8),  # Tracking every 8 frames
                    }
                )
            else:  # Default to uniform
                positions = np.linspace(start, end, num=4, dtype=int)[1:-1]

            # Collect valid frames
            for pos in sorted(set(positions)):  # Remove duplicates and sort
                if 0 <= pos < total_frames:
                    video.seek(int(pos))
                    frame = video.read()
                    if frame is not None:
                        if sampling_method == "face_detection":
                            if detect_faces(frame):
                                frames.append(frame)
                        else:
                            frames.append(frame)

        # Final fallback if we didn't get enough frames
        if len(frames) < 4:
            frames.extend(uniform_sample_frames(video, 4 - len(frames)))

        print(f"Total frames sampled: {len(frames)}")

        return frames[:32]

    except Exception as e:
        print(f"Sampling failed for {video_path}: {str(e)}")
        return uniform_sample_frames(open_video(video_path), 16)
    finally:
        pass


def detect_faces(frame):
    """Helper function for face detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    return len(faces) > 0


def process_video(video_path, capability):
    try:
        config = CAPABILITY_CONFIG.get(
            capability, CAPABILITY_CONFIG["Professional Knowledge"]
        )
        detector_config = config.get("scene_detector", {})

        scenes = detect_scenes(
            video_path,
            detector_type=detector_config["type"],
            detector_params=detector_config["params"],
            downscale_factor=detector_config["downscale"],
        )

        frames = sample_frames_from_video(
            video_path,
            scenes,
            sampling_method=detector_config["sampling"],
        )

        if not frames:
            video = open_video(video_path)
            frames = uniform_sample_frames(video, 16)

        return frames

    except Exception as e:
        print(f"Video processing failed: {str(e)}")
        return []


def uniform_sample_frames(video, num_frames):
    frames = []
    total_frames = video.duration.get_frames()
    frame_indices = np.linspace(
        0, total_frames - 1, min(num_frames, total_frames), dtype=int
    )

    for idx in frame_indices:
        video.seek(int(idx))
        frame = video.read()
        if frame is not None:
            frames.append(frame)
    return frames


df = pd.read_parquet(os.path.join(base_dir, "data", "test-00000-of-00001.parquet"))

output_csv = os.path.join(base_dir, "questions_with_answers.csv")
if not os.path.exists(output_csv):
    df.iloc[0:0].to_csv(output_csv, index=False)

for idx, row in df.iterrows():
    qid = row["qid"]
    video_id = row["video_id"]
    capability = row["capability"]
    question = row["question"]
    question_prompt = row["question_prompt"]
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    print(f"[{idx+1}/{len(df)}] Processing {video_id} with qid {qid}")

    if not os.path.exists(video_path):
        print(f"Skipping {video_id} with qid {qid}: file not found")
        continue

    try:
        keyframes = process_video(video_path, capability)

        # Prepare conversation prompt
        config = CAPABILITY_CONFIG.get(capability, {})
        full_question = (
            f"{config.get('instruction', '')}\n{question}\n{question_prompt}"
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_question},
                    {"type": "video"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs_video = processor(
            text=prompt, videos=keyframes, padding=True, return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
        response = processor.decode(output[0][2:], skip_special_tokens=True)

        clean_answer = response.split("ASSISTANT:")[-1].strip()
        df.at[idx, "answer"] = clean_answer

        # Append the current row to the CSV after processing the video
        df.iloc[[idx]].to_csv(output_csv, mode="a", header=False, index=False)

        print(f"Processed {video_id} with qid {qid}")

    except Exception as e:
        print(f"Error processing {video_id} with qid {qid}: {e}")

print("All videos processed. Answers saved to questions_with_answers.csv.")
