"""Test LLaVA-Critic-7B loading via vLLM."""
from vllm import LLM, SamplingParams
from PIL import Image

CRITIC_PROMPT = (
    "Given an image and a corresponding question, please serve as an unbiased and fair judge "
    "to evaluate the quality of the answer provided by a Large Multimodal Model (LMM).\n\n"
    "Question: {question}\n"
    "Response: {response}\n\n"
    "Please rate the response on a scale of 1 to 10, where 1 is the worst and 10 is the best.\n"
    "Your output should be in the following format:\n"
    "Score: <score>\n"
    "Reason: <reason>"
)


def main():
    model_path = "models/llava-critic-7b"
    print("Loading LLaVA-Critic via vLLM...")
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    print("Model loaded OK, class:", type(llm).__name__)

    img = Image.open("data/mm-vet/images/v1_0.png").convert("RGB")
    question = "What is x in the equation?"
    response = "x = -1 or x = -5"

    prompt_text = CRITIC_PROMPT.format(question=question, response=response)

    # Try Qwen2 chat format (since it's LlavaQwen)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    messages = [{"role": "user", "content": f"<image>\n{prompt_text}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt = {
        "prompt": text,
        "multi_modal_data": {"image": img},
    }
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm.generate([prompt], sampling_params)
    print("Output:", outputs[0].outputs[0].text[:200])
    print("TEST PASSED")


if __name__ == "__main__":
    main()
