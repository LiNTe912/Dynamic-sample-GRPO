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
    Compute reward score in a batched fashion.
    Each batch contains n_response samples.
    """
    filename = reward_kwargs['count_file']
    batch_size = reward_kwargs['batch_size']
    n_response = int(len(solution_strs) / batch_size)


    scores = []

    total = len(solution_strs)
    assert total % (batch_size * n_response) == 0, "Total size must be divisible by batch_size * n_response"

    for b in range(batch_size):
        batch_scores = []

        for i in range(n_response):
            idx = b * n_response + i
            response = solution_strs[idx]
            ground_truth = ground_truths[idx]

            extracted_response = parse_model_output(extract_xml_answer(response))
            output_words = set(extracted_response.lower().split())
            gt_list = ground_truth.split(', ')

            if "unknown" in output_words:
                batch_scores.append(0)
            elif is_ground_truth_covered(gt_list, output_words):
                batch_scores.append(1.0)
            else:
                batch_scores.append(-1.0)

        scores.extend(batch_scores)

    print(f"\nFinal Scores:\n{scores}")
    return scores

