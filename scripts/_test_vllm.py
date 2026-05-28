"""Quick test: vLLM + Qwen2.5-VL multimodal generation."""
from vllm import LLM, SamplingParams
from PIL import Image


def main():
    model_path = "models/Qwen2.5-VL-7B-Instruct"
    print("Loading model...")
    llm = LLM(
        model=model_path,
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
    )
    print("Model loaded.")

    img = Image.open("data/mm-vet/images/v1_0.png").convert("RGB")
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=128, n=2)

    prompt = {
        "prompt": (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "What is x in the equation?<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"image": img},
    }
    outputs = llm.generate([prompt], sampling_params)
    for i, o in enumerate(outputs[0].outputs):
        print(f"Sample {i+1}: {o.text[:120]}")
    print("TEST PASSED")


if __name__ == "__main__":
    main()
