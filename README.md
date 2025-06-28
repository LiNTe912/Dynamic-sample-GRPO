# Dynamic sample GRPO

Quick start
```bash
cd verl

pip install -r requirements.txt

bash examples/ds/Qwen3.sh
```

将bash文件从windows拷贝到linux后可能存在换行格式问题，解决方法
```bash
sed -i 's/\r$//' your_script.sh
```

Location for dynamic sample reward function:
"verl\verl\utils\reward_score\ds.py"

Location for normal GRPO reward function:
"verl\verl\utils\reward_score\normal.py"

Location for data:
"verl\data\nq_open"

Location for evaluation metrics:
"verl\test.py"

In Qwen3.sh:
    BACKBONE_PATH: Path to model
    REWARD_FUNCTION_PATH: Path to reward function
