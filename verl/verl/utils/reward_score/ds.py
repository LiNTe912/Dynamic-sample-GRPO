import json

def load_data_from_file(filename='count.json'):
    default_data = {
        'correct_count': 0,
        'incorrect_count': 1,
        'threshold': 0.3,
        'total_count': 1
    }

    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # 文件不存在时创建文件并写入默认内容
        with open(filename, 'w') as f:
            json.dump(default_data, f, indent=4)
        return default_data

def save_data_to_file(data, filename='count.json'):
    with open(filename, 'w') as f:
        json.dump(data, f)


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def parse_model_output(output: str) -> str:
    # 去掉空格和换行符，并将其转换为小写进行统一处理
    cleaned_output = output.strip().lower()

    # 判断是否输出了 "Unknown"
    if cleaned_output == "unknown":
        return "unknown"  # 如果是 Unknown，就返回 "Unknown"
    else:
        return output  # 如果不是 "Unknown"，返回原始输出


def is_ground_truth_covered(ground_truth: list, output_words: set) -> bool:
    for truth in ground_truth:
        answer_words = set(truth.lower().split())
        if answer_words.issubset(output_words) or output_words.issubset(answer_words):
            return True
    return False

def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    filename = reward_kwargs['count_file']
    data = load_data_from_file(filename)
    scores = []

    has_knowledge = False

    for i, response in enumerate(solution_strs):
        extracted_response = parse_model_output(extract_xml_answer(response))
        # extracted_response_list.append(extracted_response)
        
        output_words = set(extracted_response.lower().split())
        ground_truth = ground_truths[i].split(', ')
        
        if "unknown" in output_words:
            scores.append(0)
        elif is_ground_truth_covered(ground_truth, output_words):
            has_knowledge = True
            scores.append(1.0)
        else:
            scores.append(-1.0)
        
        print(f"\nResponse:\n{response}", f"\nExtracted:\n{extracted_response}", f"\nAnswer:\n{ground_truths[i]}")

    if (data["incorrect_count"]) / (data["correct_count"] + data["incorrect_count"]) > data["threshold"] and not has_knowledge:
        scores = [0 for _ in solution_strs]
    
    else:
        if has_knowledge: 
            data["correct_count"] += 1
        else:
            data["incorrect_count"] += 1
    
    data["total_count"] += 1

    save_data_to_file(data, filename)

    print(f"Scores:\n{scores} ")

    return scores
