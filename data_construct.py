import os
import pandas as pd
import torch
import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa
from qwen_vl_utils import process_vision_info

data_path = "your path/iemocap_test.csv"
data = pd.read_csv(data_path)
data = data.dropna(subset=['video_id', 'text', 'path'])

output_json_path = "test_dataset_extract_1_17.json"

vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "your path/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

vl_processor = AutoProcessor.from_pretrained("your path/qwen2_vl_lora_sft")

audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "your path/Qwen2-Audio-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto"
)
audio_processor = AutoProcessor.from_pretrained("your path/Qwen2-Audio-7B-Instruct")

if os.path.exists(output_json_path):
    with open(output_json_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
else:
    processed_data = []

processed_ids = {sample['input'].split("；")[0].split("Video：")[1] for sample in processed_data}

for _, row in data.iterrows():
    video_id = row['video_id']
    if video_id in processed_ids:
        continue  

    try:
        text = row['text']
        actual_label = row['emotion']
        path = row['path']

        audio_path = f"your path/{path}"
        video_path = f"your path/output_slices/DivX/{video_id}.avi"

        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            print(f"File does not exist, skipping: {video_id}")
            continue

        conversation_audio = [
            {'role': 'system', 'content': "You are an emotional intelligence expert."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": (
                     f"Analyze the emotion of the speaker based on the tone of the audio. You can also refer to the text modality: {text}."
                )}
            ]}
        ]
        text_prompt_audio = audio_processor.apply_chat_template(conversation_audio, add_generation_prompt=True)

        audios = []
        for message in conversation_audio:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audio_data, _ = librosa.load(
                            ele['audio_url'],
                            sr=audio_processor.feature_extractor.sampling_rate
                        )
                        audios.append(audio_data)

        audio_inputs = audio_processor(text=[text_prompt_audio], audios=audios, padding=True, return_tensors="pt")
        audio_inputs = audio_inputs.to("cuda")

        audio_output_text = ""
        try:
            audio_output_ids = audio_model.generate(**audio_inputs, max_new_tokens=128)
            audio_generated_ids = audio_output_ids[:, audio_inputs.input_ids.size(1):]
            audio_output_text = audio_processor.batch_decode(audio_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        except Exception as e:
            print(f"Error processing audio (ID: {video_id}): {e}")
            del audio_inputs
            torch.cuda.empty_cache()
            continue

        conversation_video = [
            {'role': 'system', 'content': "As an expert in sentiment analysis, you are tasked with extracting emotional features from the video depicting the speaker's environment."},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
                {"type": "text", "text": (
                    f"Analyze the speaker's emotion based on the facial expressions. You can also refer to the text modality: {text}."
                )}
            ]}
        ]
        text_prompt_video = vl_processor.apply_chat_template(conversation_video, add_generation_prompt=True)

        video_output_text = ""
        try:
            image_inputs, video_inputs = process_vision_info(conversation_video)
            video_inputs = vl_processor(
                text=[text_prompt_video],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            video_inputs = video_inputs.to("cuda")
            output_ids_video = vl_model.generate(**video_inputs, max_new_tokens=128)
            generated_ids_video = output_ids_video[:, video_inputs.input_ids.size(1):]
            video_output_text = vl_processor.batch_decode(generated_ids_video, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        except Exception as e:
            print(f"Error processing video (ID: {video_id}): {e}")
            continue

        sample_data = {
            "instruction": (
                "You are an emotional intelligence expert tasked with determining the sentiment. Consider all given modalities (Video is the scene of the speaker, text is the content of the speaker, audio is the tone of the speaker and so on). Use the text modality to interpret the other two modalities step by step. Consider the consistency or conflict between these modalities; if there is a conflict, integrating audio and visual cues to resolve ambiguities. Conduct a comprehensive assessment of the overall emotion and respond with a single word: Negative or Non-Negative."
            ),
            "input": f"Video：{video_output_text}；Text：{text}；Audio：{audio_output_text}",
            "output": actual_label
        }

        processed_data.append(sample_data)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"Error processing sample (ID: {video_id}): {e}")
        continue

print(f"All samples processed. Dataset saved to {output_json_path}")
