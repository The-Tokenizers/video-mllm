import os

import cv2
import numpy as np
import pandas as pd
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
from scipy.signal import find_peaks

from capability_config import CAPABILITY_CONFIG
from frame_extractor import (
    detect_face_frames,
    extract_action_frames,
    multi_object_tracking_frames,
    track_motion_frames,
    uniform_sample_frames,
)

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(base_dir, "models", "llava_next_video_7b_dpo_model")
processor_dir = os.path.join(model_dir, "processor")
model_subdir = os.path.join(model_dir, "model")
video_dir = os.path.join(base_dir, "data", "Benchmark-AllVideos-HQ-Encoded-challenge")

processor = LlavaNextVideoProcessor.from_pretrained(processor_dir, use_fast=True)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_subdir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def get_capability_config(capability):
    return CAPABILITY_CONFIG.get(
        capability,
        {
            "instruction": "",
            "frame_selection_strategy": "uniform_sample_frames",
        },
    )


def read_video_and_sample_frames(
    video_path, frame_selection_strategy="uniform_sample_frames", num_frames=16
):
    """Read video and extract frames using the specified strategy."""
    if frame_selection_strategy == "uniform_sample_frames":
        print("Using uniform sampling strategy")
        return uniform_sample_frames(video_path, num_frames)
    elif frame_selection_strategy == "extract_action_frames":
        print("Using action extraction strategy")
        return extract_action_frames(video_path, num_frames)
    elif frame_selection_strategy == "detect_face_frames":
        print("Using face detection strategy")
        return detect_face_frames(video_path, num_frames)
    elif frame_selection_strategy == "track_motion_frames":
        print("Using motion tracking strategy")
        return track_motion_frames(video_path, num_frames)
    elif frame_selection_strategy == "multi_object_tracking_frames":
        print("Using multi-object tracking strategy")
        return multi_object_tracking_frames(video_path, num_frames)
    else:
        raise ValueError(f"Unknown frame strategy: {frame_selection_strategy}")


df = pd.read_parquet(os.path.join(base_dir, "data", "test-00000-of-00001.parquet"))

output_csv = os.path.join(base_dir, "questions_with_answers.csv")
if not os.path.exists(output_csv):
    df.iloc[0:0].to_csv(output_csv, index=False)

for idx, row in df.iterrows():
    qid = row["qid"]
    video_id = row["video_id"]
    capability = row["capability"]
    capability_config = get_capability_config(capability)
    question = row["question"]
    question_prompt = row["question_prompt"]
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    print(f"[{idx+1}/{len(df)}] Processing {video_id} with qid {qid}")

    if not os.path.exists(video_path):
        print(f"Skipping {video_id} with qid {qid}: file not found")
        continue

    try:
        frame_selection_strategy = capability_config["frame_selection_strategy"]
        keyframes = read_video_and_sample_frames(
            video_path, frame_selection_strategy, num_frames=16
        )

        # Prepare conversation prompt
        full_question = (
            f"{capability_config['instruction']}\n{question}\n{question_prompt}"
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
