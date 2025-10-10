import json
import csv
import random
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset, Dataset
import re
import ast

# 设置随机种子保证可重复性
random.seed(42)

def load_custom_dataset(file_path="/home/zwt/Exp/data/hotpotQA/test.json", max_line=2000, starting_point=1) -> Dataset:
    data = []
    i = 0
    with open(file_path, "r", encoding="utf-8") as f:
        # reader = csv.DictReader(f)          # 自动用第一行当键
        # data = [
        #     {"question": row["problem"].strip(),
        #     "answer":  row["answer"].strip()}
        #     for row in reader
        # ]

        data = json.load(f)

        # for line in f:
        #     line = line.strip()
        #     if line:
        #         data.append(json.loads(line))

    processed_data = []
    for item in data:
        if i >= max_line + starting_point:
            break      
        i += 1
        if i <= starting_point: 
            continue

        # answer = ast.literal_eval(item["answer"])
        answer = item["answer"]
        messages = item["question"]
        
        prompt = [
            {"role": "system", "content": "\nRespond in the following format:\n<think>\n...\n</think>\n<answer>\n...\n</answer>\n"},
            {"role": "user", "content": f"Answer the following question with one or few words. If you do not know the answer, answer 'Unknown'. Question: {messages}\n"},
        ]

        processed_data.append({"prompt": prompt, "answer": answer, "question": messages})

    return Dataset.from_list(processed_data)


# 加载训练好的模型
# 初始化模型
model_path = "/data2/wentao/qwen2.5-7b"

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",  # 可根据你的 GPU 支持情况选择 "float16" / "bfloat16"
    tensor_parallel_size=2,
    enable_lora=True,
    max_lora_rank=64,
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
    max_tokens=512,
    stop=["</answer>"]
)

lora_request = LoRARequest(
    lora_path="/data2/zwt/verl/checkpoints/DS-GRPO-qwen2.5-7b_hotpotQA-first10k/global_step_312/actor/lora_adapter",
    lora_name="qwen3-8b-instruct",
    lora_int_id=1)

def generate_response(dataset):
    messages_list = dataset["prompt"]
    outputs = llm.chat(
        messages_list,
        sampling_params=sampling_params,
        lora_request=None,
        #chat_template_kwargs={"enable_thinking": True},  # 若要启用 <think> 结构，则设置为 True
    )
    return outputs

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def parse_model_output(output: str) -> str:
    # 去掉空格和换行符，并将其转换为小写进行统一处理
    cleaned_output = re.sub(r'[^a-zA-Z0-9 ]', '', output)
    cleaned_output = cleaned_output.strip().lower()

    # 判断是否输出了 "Unknown"
    if cleaned_output == "unknown":
        return "unknown"  # 如果是 Unknown，就返回 "Unknown"
    else:
        return cleaned_output  # 如果不是 "Unknown"，返回原始输出


from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
STOPWORDS -= {"yes", "no", "not", "true", "false"}

def is_ground_truth_covered(truth: str, output_words: set) -> bool: 
    # 去除停用词
    output_filtered = {w for w in output_words if w not in STOPWORDS}

    truth = re.sub(r'[^a-zA-Z0-9 ]', '', truth)
    answer_words = {w for w in truth.lower().split() if w not in STOPWORDS}

    if not answer_words and not output_filtered:
        return True

    # 比较子集关系
    if answer_words.issubset(output_filtered) or output_filtered.issubset(answer_words):
        return True
    
    return False


# 2025.10.7 追加LLM评估代码
from openai import OpenAI

with open("api_keys.json", "r") as f:
    api_keys = json.load(f)["deepseek"]

def LLM_as_judge(question, ground_truth, extracted):

    global api_keys

    client = OpenAI(
        api_key=api_keys,
        base_url="https://api.deepseek.com")
    
    JUDGE_SYSTEM_PROMPT = (
    "You are a strict grading assistant. "
    "Given a question, the model's answer, and the gold reference answers (list), "
    "decide if the model's answer is semantically correct. "
    "Be tolerant of minor wording differences, but do not accept contradictions or fabrications. "
    "If the ground truths are multiple, treat the answer as correct if it matches ANY of them.\n"
    "Return ONLY with 'Correct' or 'Incorrect'"
    )

    prompt = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nGround Truth Answers: {', '.join(ground_truth)}\nModel Answer: {extracted}\n\nPlease evaluate the correctness of the extracted answer. If the answer is correct, respond with 'Correct'. If it is incorrect or if you are unsure, respond with 'Incorrect'."}
    ]
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=prompt,
                stream=False,
                max_tokens=1024,
                temperature=0.0,
            )
            judge_text = response.choices[0].message.content.strip().lower()
            return judge_text
        except Exception as e:
            print(f"Error during LLM judging (attempt {attempt + 1}/5): {e}")
    return "incorrect"  # 如果多次尝试都失败，默认返回 "incorrect


test_dataset = load_custom_dataset()
certainty_messages_list = []
answer_list = []
sure_count = 0
true_count = 0
outputs = generate_response(test_dataset)

# ==================  追加代码开始  ==================
import os
import jsonlines

details_file = "inference_details.jsonl"
sure_count = 0
true_count = 0
llm_judge_true_count = 0

with jsonlines.open(details_file, mode="w") as writer:
    for i, output in enumerate(outputs):
        question = test_dataset[i]["question"]
        ground_truth = test_dataset[i]["answer"]          # list
        model_raw = output.outputs[0].text                # 原始模型返回
        extracted = parse_model_output(extract_xml_answer(model_raw))

        has_knowledge = "unknown" not in extracted.lower()
        rule_correct = False
        llm_judge_correct = False

        if has_knowledge:
            sure_count += 1
            output_words = set(extracted.lower().split())
            if is_ground_truth_covered(ground_truth, output_words) and len(output_words) > 0:
                true_count += 1
                rule_correct = True
            judge_output = LLM_as_judge(question, ground_truth, extracted)
            if judge_output == 'correct':
                llm_judge_true_count += 1
                llm_judge_correct = True

        writer.write({
            "question": question,
            "ground_truth": ground_truth,
            "model_output": model_raw,
            "extracted_answer": extracted,
            "has_knowledge": has_knowledge,
            "rule_correct": rule_correct,
            "llm_judge_correct": llm_judge_correct 
        })

# 用更新后的 sure_count / true_count 计算指标
precision = true_count / sure_count if sure_count else 0
recall = true_count / len(test_dataset) if test_dataset else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

llm_judge_precision = llm_judge_true_count / sure_count if sure_count else 0
llm_judge_recall = llm_judge_true_count / len(test_dataset) if test_dataset else 0
llm_judge_f1 = 2 * llm_judge_precision * llm_judge_recall / (llm_judge_precision + llm_judge_recall) if (llm_judge_precision + llm_judge_recall) else 0

print("\n评估结果统计:")
print(f"总样本数: {len(test_dataset)}")
print(f"模型判断为确定的样本数: {sure_count}")
print(f"Rule: 模型判断为确定且正确的样本数: {true_count}")
print(f"Rule: 模型确定性判断准确率: {precision * 100:.2f}%")
print(f"Rule: 模型确定性判断召回率: {recall * 100:.2f}%")
print(f"Rule: 模型F1分数: {f1 * 100:.2f}%")
print(f"LLM Judge: 模型判断为确定且正确的样本数: {llm_judge_true_count}")
print(f"LLM Judge: 模型确定性判断准确率: {llm_judge_precision * 100:.2f}%")
print(f"LLM Judge: 模型确定性判断召回率: {llm_judge_recall * 100:.2f}%")
print(f"LLM Judge: 模型F1分数: {llm_judge_f1 * 100:.2f}%")
print(f"详细结果已写入: {os.path.abspath(details_file)}")
# ==================  追加代码结束  ==================

