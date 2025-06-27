from datasets import load_dataset, Dataset
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 设置系统提示词
SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

# 自定义数据加载函数
def load_custom_dataset(file_path="/dev_data/wy/zwt/myKnowledgeBoundaries/nq_open/NQ-open.efficientqa.test.1.1.jsonl", max_line=5000, starting_point=1) -> Dataset:
    data = []
    i = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    processed_data = []
    for item in data:
        if i >= max_line + starting_point:
            break      
        i += 1
        if i <= starting_point: 
            continue

        answer = item["answer"]
        messages = item["question"]
        
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Answer the following question based on your internal knowledge, the length for reasoning should be less than 500 words, and the answer between <answer> and </answer> should be one or few words. If you are not sure, you can answer \"Unknown\". Question: {messages}\n"},
        ]

        processed_data.append({"prompt": prompt, "answer": answer})

    return Dataset.from_list(processed_data)

# 加载测试数据
test_dataset = load_custom_dataset(max_line=10, starting_point=0)
print("Test data loaded:", len(test_dataset))

# 初始化模型
model_path = "/dev_data/wy/zwt/qwen3-4b"
lora_path = "/dev_data/wy/zwt/verl/checkpoints/DS-verl/DS-GRPO-Qwen3-4B/0627/DS-GRPO-grpo-160850/global_step_100/actor/lora_adapter"

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",  # 可根据你的 GPU 支持情况选择 "float16" / "bfloat16"
    tensor_parallel_size=1,
    enable_lora=True,
    max_lora_rank=64
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=["</answer>"]
)

# 准备输入和 LoRA 请求
prompts = []
for idx, sample in enumerate(test_dataset):
    prompt_text = ""
    for message in sample["prompt"]:
        prompt_text += f"{message['content']}\n"
    prompts.append(prompt_text)

# 构造 LoRA 请求（假设 adapter_name 就叫 'verl_adapter'）
adapter_name = "verl_adapter"
lora_req = LoRARequest(
    lora_int_id=1,
    lora_name=adapter_name,
    lora_path=lora_path
)

# 运行模型推理
outputs = llm.generate(prompts, sampling_params=sampling_params, lora_request=lora_req)

# 打印输出示例
for i, output in enumerate(outputs[:5]):  # 只显示前5条
    print(f"\nExample {i + 1}")
    print("Prompt:", prompts[i])
    print("Generated output:", output.outputs[0].text)
