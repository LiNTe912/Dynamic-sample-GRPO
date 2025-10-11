from ds import compute_score
def test_compute_score():
    # 模拟输入数据
    data_sources = None
    extra_infos = None

    # ground truth 与模型输出
    ground_truths = ["paris"]*10
    solution_strs = [
        "<think>I recall...</think>{\"answer\": \"Paris\", \"confidence\": \"20%\"}",
        "<think>I recall...</think><answer>{\"answer\": \"Berlin\", \"confidence\": \"70%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Tokyo\", \"confidence\": \"10%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Unknown\", \"confidence\": \"20%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Rome\", \"confidence\": \"60%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Rome\", \"confidence\": \"60%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Paris\", \"confidence\": \"20%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Berlin\", \"confidence\": \"70%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Unknown\", \"confidence\": \"20%\"}</answer>",
        "<think>I recall...</think><answer>{\"answer\": \"Tokyo\", \"confidence\": \"20%\"}</answer>",
    ]

    # 模拟参数
    reward_kwargs = {
        "count_file": "mock.json",
        "n_samples": 10,  # 每批10个response
    }

    # 调用 compute_score
    scores = compute_score(data_sources, solution_strs, ground_truths, extra_infos, **reward_kwargs)

    # 打印结果
    print("\n=== Final Computed Scores ===")
    for i, s in enumerate(scores):
        print(f"Sample {i}: {s}")


# =================== 执行测试 ===================

if __name__ == "__main__":
    test_compute_score()