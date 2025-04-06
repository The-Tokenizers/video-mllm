import os
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_dir = os.path.join(base_dir, "models", "llava_next_video_7b_dpo_model")
processor_dir = os.path.join(model_dir, "processor")
model_subdir = os.path.join(model_dir, "model")

os.makedirs(processor_dir, exist_ok=True)
os.makedirs(model_subdir, exist_ok=True)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-DPO-hf", use_fast=True)
model = LlavaNextVideoForConditionalGeneration.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-DPO-hf", device_map="auto")

processor.save_pretrained(processor_dir)
model.save_pretrained(model_subdir, safe_serialization=True)
