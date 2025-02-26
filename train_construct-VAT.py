import os
import json
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

data_path = "your data_path"
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

output_json_path = "train_dataset_text_with_reasoning.json"

model_name = "your path"
print("Starting model loading...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="/tmp/offload",
    offload_state_dict=True
)
print("Model loaded successfully!")

tokenizer = AutoTokenizer.from_pretrained(model_name)

if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
else:
    processed_data = []

processed_ids = {sample['id'] for sample in processed_data}

for idx, sample in enumerate(data):
    video_id = f"sample_{idx}"
    if video_id in processed_ids:
        continue  

    try:
        text = sample['input']
        actual_label = sample['output']
        incorrect_label = f"not {actual_label}"

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

        reasoning_chains = {}
        for chain_type, prompt in reasoning_templates.items():
            try:
                model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

                generated_ids = model.generate(
                    **model_inputs,
                    temperature=0.7,
                    top_p=0.9
                )

                generated_text = tokenizer.batch_decode(
                    generated_ids[:, model_inputs.input_ids.size(1):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                reasoning_chains[chain_type] = generated_text
            except Exception as e:
                print(f"Error generating reasoning chain (ID: {video_id}, {chain_type}): {e}")
                reasoning_chains[chain_type] = f"Error generating reasoning chain: {e}"
                continue

        selection_prompt = (
            "You are an emotional intelligence expert. Based on the provided multimodal information, select the most accurate reasoning chain:\n\n"
            f"1. {reasoning_chains.get('Neutral Reasoning Chain', 'Error')}\n\n"
            f"2. {reasoning_chains.get('Incorrectly Labeled Chain', 'Error')}\n\n"
            f"3. {reasoning_chains.get('Correctly Labeled Chain', 'Error')}\n\n"
            "Please select the chain (1, 2, or 3) that most accurately aligns with the provided text."
        )

        try:
            model_inputs = tokenizer([selection_prompt], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                temperature=0.7,
                top_p=0.9
            )

            selected_chain = tokenizer.batch_decode(
                generated_ids[:, model_inputs.input_ids.size(1):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
        except Exception as e:
            print(f"Error selecting reasoning chain (ID: {video_id}): {e}")
            selected_chain = f"Error selecting reasoning chain: {e}"

        sample_data = {
            "id": video_id,
            "instruction": "Determine the most accurate reasoning chain from the three options.",
            "text": text,
            "actual_label": actual_label,
            "reasoning_chains": reasoning_chains,
            "selected_chain": selected_chain
        }

        processed_data.append(sample_data)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Error processing sample (ID: {video_id}): {e}")
        continue

print(f"All samples processed. Dataset saved to {output_json_path}")
