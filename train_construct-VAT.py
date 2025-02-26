import os
import json
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 数据路径
data_path = "/nfs/home/9105_zengjiandian/JJ327/LLM+MSA/MOSI/train_dataset.json"
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 定义输出JSON文件路径
output_json_path = "train_dataset_text_with_reasoning.json"

# 加载模型和分词器
model_name = "/nfs/home/9105_zengjiandian/JJ327/MSA/hub/qwen/Qwen2.5-72B-Instruct"
print("开始加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="/tmp/offload",
    offload_state_dict=True
)
print("模型加载完成！")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 如果JSON文件已经存在，加载已处理的样本
if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
else:
    processed_data = []

# 提取已处理的索引，避免重复处理
processed_ids = {sample['id'] for sample in processed_data}

# 遍历未处理的样本
for idx, sample in enumerate(data):
    # 使用索引值作为唯一ID
    video_id = f"sample_{idx}"
    if video_id in processed_ids:
        continue  # 跳过已处理的样本

    try:
        # 提取文本和标签
        text = sample['input']
        actual_label = sample['output']
        incorrect_label = f"not {actual_label}"

        # 定义推理链模板
        reasoning_templates = {
            "Neutral Reasoning Chain": (
                "You are an emotional intelligence expert tasked with determining the emotion based on the following categories: Anger (ang), Excitement (exc), Frustration (fru), Happiness (hap), Neutral (neu), Surprise (sur), and Sadness (sad). \n"
                f"{text}\n Please conduct a comprehensive assessment of the overall emotion and provide a reasoning chain."
            ),
            "Incorrectly Labeled Chain": (
                "You are an emotional intelligence expert. Construct a reasoning chain that supports an intentionally incorrect label.\n"
                f"{text}\nThe correct label is '{incorrect_label}'. Please construct a reasoning chain that aligns with this label."
            ),
            "Correctly Labeled Chain": (
                "You are an emotional intelligence expert. Construct a reasoning chain that supports the correct label.\n"
                f"{text}\nThe correct label is '{actual_label}'. Please construct a reasoning chain that aligns with this label."
            )
        }

        # 存储三种推理链的结果
        reasoning_chains = {}
        for chain_type, prompt in reasoning_templates.items():
            try:
                # 分词器生成输入
                model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

                # 推理生成结果
                generated_ids = model.generate(
                    **model_inputs,
                    temperature=0.7,
                    top_p=0.9
                )

                # 解码生成的文本
                generated_text = tokenizer.batch_decode(
                    generated_ids[:, model_inputs.input_ids.size(1):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                reasoning_chains[chain_type] = generated_text
            except Exception as e:
                print(f"生成推理链时出错（ID: {video_id}, {chain_type}）: {e}")
                reasoning_chains[chain_type] = f"Error generating reasoning chain: {e}"
                continue

        # 构造模型选择推理链的模板
        selection_prompt = (
            "You are an emotional intelligence expert. Based on the provided multimodal information, select the most accurate reasoning chain:\n\n"
            f"1. {reasoning_chains.get('Neutral Reasoning Chain', 'Error')}\n\n"
            f"2. {reasoning_chains.get('Incorrectly Labeled Chain', 'Error')}\n\n"
            f"3. {reasoning_chains.get('Correctly Labeled Chain', 'Error')}\n\n"
            "Please select the chain (1, 2, or 3) that most accurately aligns with the provided text."
        )

        try:
            # 分词器生成选择输入
            model_inputs = tokenizer([selection_prompt], return_tensors="pt").to(model.device)

            # 推理生成结果
            generated_ids = model.generate(
                **model_inputs,
                temperature=0.7,
                top_p=0.9
            )

            # 解码选择结果
            selected_chain = tokenizer.batch_decode(
                generated_ids[:, model_inputs.input_ids.size(1):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        except Exception as e:
            print(f"选择推理链时出错（ID: {video_id}）: {e}")
            selected_chain = f"Error selecting reasoning chain: {e}"

        # 构建JSON数据
        sample_data = {
            "id": video_id,
            "instruction": "Determine the most accurate reasoning chain from the three options.",
            "text": text,
            "actual_label": actual_label,
            "reasoning_chains": reasoning_chains,
            "selected_chain": selected_chain
        }

        # 添加到数据集中并保存
        processed_data.append(sample_data)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"处理样本时出错（ID: {video_id}）: {e}")
        continue

print(f"所有样本处理完成，数据集已保存至 {output_json_path}")
