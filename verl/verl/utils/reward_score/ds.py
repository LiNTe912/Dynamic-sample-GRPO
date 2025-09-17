import json
import re
from collections import defaultdict

def sortQA(questions, answers):
    # 1. 配对：打包成 (question, answer, original_index) 三元组
    paired = list(zip(questions, answers, range(len(questions))))

    # 2. 分组：将相同的问题聚到一起
    grouped = defaultdict(list)
    for q, a, idx in paired:
        grouped[q].append((q, a, idx))  # 保留原顺序

    # 3. 排序：将每组拼接回来
    sorted_triplets = []
    for q in sorted(grouped.keys()):
        sorted_triplets.extend(grouped[q])

    # 4. 解包为三个列表：问题、答案、新→原的映射
    sorted_questions, sorted_answers, original_indices = zip(*sorted_triplets)
    sorted_questions = list(sorted_questions)
    sorted_answers = list(sorted_answers)
    original_indices = list(original_indices)  # 这是映射：sorted -> original

    return sorted_questions, sorted_answers, original_indices

def load_data_from_file(filename='count.json'):
    default_data = {
        'correct_count': 1,
        'incorrect_count': 0,
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
    cleaned_output = re.sub(r'[^a-zA-Z0-9 ]', '', output)
    cleaned_output = cleaned_output.strip().lower()

    # 判断是否输出了 "Unknown"
    if cleaned_output == "unknown":
        return "unknown"  # 如果是 Unknown，就返回 "Unknown"
    else:
        return output  # 如果不是 "Unknown"，返回原始输出

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def is_ground_truth_covered(ground_truth: list, output_words: set) -> bool: 
    # 去除停用词
    output_filtered = {w for w in output_words if w not in STOPWORDS}
    if not output_filtered:
        return False

    for truth in ground_truth:
        truth = re.sub(r'[^a-zA-Z0-9 ]', '', truth)
        answer_words = {w for w in truth.lower().split() if w not in STOPWORDS}

        if not answer_words:
            continue

        # 比较子集关系
        if answer_words.issubset(output_filtered) or output_filtered.issubset(answer_words):
            return True
    return False


# def jaccard_similarity(set1, set2):
#     return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

# def is_ground_truth_covered(ground_truth: list, output_words: set, threshold=0.5) -> bool:
#     for truth in ground_truth:
#         truth = re.sub(r'[^a-zA-Z0-9 ]', '', truth)
#         answer_words = set(truth.lower().split())

#         if not answer_words:
#             continue

#         sim = jaccard_similarity(answer_words, output_words)
#         if sim >= threshold:
#             return True
#     return False


def compute_score(data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs):
    filename = reward_kwargs['count_file']
    n_response = reward_kwargs['n_samples']
    batch_size = int(len(solution_strs) / n_response)
    data = load_data_from_file(filename)

    ground_truths, solution_strs, index_map = sortQA(ground_truths, solution_strs)

    extracted_response_list = []
    semantic_scores = []
    format_scores = []

    total = len(solution_strs)
    assert total % (batch_size * n_response) == 0, "Total size must be divisible by batch_size * n_response"

    # 记录需要在循环结束后用均值回填的位置
    flagged_spans = []  # 每项为 (start_idx_in_semantic_scores, length)

    for b in range(batch_size):
        has_knowledge = 0
        batch_semantic_scores = []
        batch_format_scores = []

        for i in range(n_response):
            idx = b * n_response + i
            response = solution_strs[idx]
            ground_truth = ground_truths[idx]

            extracted_response = parse_model_output(extract_xml_answer(response))
            extracted_response_list.append(extracted_response)

            output_words = set(extracted_response.lower().split())
            gt_list = ground_truth.split(', ')
            print(f"Ground truth list: {gt_list}")

            # 计算语义得分
            if "unknown" in output_words:
                batch_semantic_scores.append(0.0)
            elif is_ground_truth_covered(gt_list, output_words) and len(output_words) > 0:
                has_knowledge += 1
                batch_semantic_scores.append(1.0)
            else:
                batch_semantic_scores.append(-1.0)

            # batch_format_scores.append(0.0)
            # 计算格式得分
            if all(tag in response for tag in ['<think>', '</think>', '<answer>', '</answer>']):
            # if len(extracted_response.lower().split()) < 20:
                batch_format_scores.append(0.0)
            else:
                batch_format_scores.append(-1.0)

        print(f"Has knowledge: {has_knowledge}")

        # 惩罚机制：整批设为0
        # 检查batch_format_scores是否所有项都相同

        all_scores_same = len(set(batch_semantic_scores)) == 1 if batch_semantic_scores else True

        print(f"Batch raw score: {batch_semantic_scores}")
        
        if all_scores_same or ((data["incorrect_count"]) / (data["correct_count"] + data["incorrect_count"]) > data["threshold"] and has_knowledge == 0):
            start_pos = len(semantic_scores)
            flagged_spans.append((start_pos, n_response))
        else:
            if has_knowledge > 0:
                data["correct_count"] += 1
            else:
                data["incorrect_count"] += 1
        data["total_count"] += 1

        semantic_scores.extend(batch_semantic_scores)
        format_scores.extend(batch_format_scores)

    save_data_to_file(data, filename)

    # ========== 关键修改开始 ==========
    # 1. 先把所有被标记的索引收集起来（方便后续过滤）
    flagged_indices = set()
    for start, length in flagged_spans:
        flagged_indices.update(range(start, start + length))

    # 2. 只拿未标记样本计算均值
    unflagged_scores = [s for idx, s in enumerate(semantic_scores)
                        if idx not in flagged_indices]
    if unflagged_scores:               # 防止除零
        mean_sem = sum(unflagged_scores) / len(unflagged_scores)
    else:
        mean_sem = 0.0                 # 极端情况：全部样本都被标记

    # 3. 回填
    for start, length in flagged_spans:
        for j in range(start, start + length):
            semantic_scores[j] = mean_sem
    # ========== 关键修改结束 ==========

    # 合并分数并恢复顺序
    total_scores = [s + f for s, f in zip(semantic_scores, format_scores)]

    recovered_score = [None] * len(total_scores)
    for i, original_idx in enumerate(index_map):
        recovered_score[original_idx] = total_scores[i]

    print(f"Extracted answers: {extracted_response_list}")
    print(f"\nRecovered Scores (semantic + format):\n{recovered_score}")

    return recovered_score
