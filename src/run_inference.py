import os

import numpy as np
import pandas as pd
import av
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(base_dir, "models", "llava_next_video_7b_dpo_model")
processor_dir = os.path.join(model_dir, "processor")
model_subdir = os.path.join(model_dir, "model")
video_dir = os.path.join(base_dir, "data", "Benchmark-AllVideos-HQ-Encoded-challenge")

processor = LlavaNextVideoProcessor.from_pretrained(processor_dir, use_fast=True)
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    model_subdir,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

def read_video_pyav(container, indices):
    """Decode selected frames from a video using PyAV."""
    frames = []
    container.seek(0)
    frame_idx_set = set(indices)
    for i, frame in enumerate(container.decode(video=0)):
        if i > indices[-1]:
            break
        if i in frame_idx_set:
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)

df = pd.read_parquet(os.path.join(base_dir, "data", "test-00000-of-00001.parquet"))

output_csv = os.path.join(base_dir, "questions_with_answers.csv")
if not os.path.exists(output_csv):
    df.iloc[0:0].to_csv(output_csv, index=False)

for idx, row in df.iterrows():
    qid = row["qid"]
    video_id = row["video_id"]
    question = row["question"]
    question_prompt = row["question_prompt"]
    video_path = os.path.join(video_dir, f"{video_id}.mp4")

    print(f"[{idx+1}/{len(df)}] Processing {video_id} with qid {qid}")

    if not os.path.exists(video_path):
        print(f"Skipping {video_id} with qid {qid}: file not found")
        continue

    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, num=16, dtype=int)
        print(f"Total frames in {video_id}: {total_frames}")
        print(f"Total frames sampled: {len(indices)}")

        clip = read_video_pyav(container, indices)

        # Prepare conversation prompt
        full_question = f"{question}\n{question_prompt}"
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
        inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
        response = processor.decode(output[0][2:], skip_special_tokens=True)

        clean_answer = response.split("ASSISTANT:")[-1].strip()
        df.at[idx, "answer"] = clean_answer

        # Append the current row to the CSV after processing the video
        df.iloc[[idx]].to_csv(output_csv, mode='a', header=False, index=False)

        print(f"Processed {video_id} with qid {qid}")

    except Exception as e:
        print(f"Error processing {video_id} with qid {qid}: {e}")

print("All videos processed. Answers saved to questions_with_answers.csv.")
