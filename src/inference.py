"""VC-Inspector inference script.

Usage:
    python -m src.inference \
        --model dipta007/VCInspector-7B \
        --video path/to/video.mp4 \
        --caption "A man is playing guitar in a field"
"""

import argparse

import numpy as np
from decord import VideoReader, cpu
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

NUM_FRAMES = 32

EVAL_PROMPT = """{image_tags}
<caption>{caption}</caption>

You are given a video and a caption describing the video content. Please rate the helpfulness, relevance, accuracy, level of details of the caption. The overall score should be on a scale of 1 to 5, where a higher score indicates better overall performance. Please first output a single line containing only one integer indicating the score. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias. STRICTLY FOLLOW THE FORMAT."""


def extract_frames(video_path: str, num_frames: int = NUM_FRAMES) -> list:
    """Extract evenly-spaced frames from a video."""
    vr = VideoReader(video_path, ctx=cpu(0))
    indices = np.linspace(0, len(vr) - 1, num_frames, dtype=int)
    return [vr[i].asnumpy() for i in indices]


def evaluate(
    model_id: str,
    video_path: str,
    caption: str,
    num_frames: int = NUM_FRAMES,
    max_new_tokens: int = 1024,
) -> str:
    """Score a video caption using VC-Inspector.

    Returns the model output (score on first line, explanation on subsequent lines).
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    frames = extract_frames(video_path, num_frames)
    image_tags = "<image>" * num_frames
    prompt = EVAL_PROMPT.format(image_tags=image_tags, caption=caption)

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": f} for f in frames]
            + [{"type": "text", "text": prompt}],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="VC-Inspector: Video Caption Evaluation")
    parser.add_argument("--model", type=str, default="dipta007/VCInspector-7B",
                        help="Model ID (default: dipta007/VCInspector-7B)")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--caption", type=str, required=True, help="Caption to evaluate")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES,
                        help="Number of frames to extract (default: 32)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens to generate (default: 1024)")
    args = parser.parse_args()

    output = evaluate(args.model, args.video, args.caption, args.num_frames, args.max_new_tokens)
    print(output)


if __name__ == "__main__":
    main()
