import os
import re
import pandas as pd
from tqdm import tqdm
from typing import List
from dataclasses import dataclass
from msa_llm import DialogueModelAPI

@dataclass
class ModalityData:
    text: str
    visual: str
    audio: str

    @classmethod
    def from_input_string(cls, input_str: str) -> 'ModalityData':
        video_match = re.search(r"Video：(.+?)；Text：", input_str)
        text_match = re.search(r"Text：(.+?)；Audio：", input_str)
        audio_match = re.search(r"Audio：(.*)", input_str)

        video = video_match.group(1).strip() if video_match else ""
        text = text_match.group(1).strip() if text_match else ""
        audio = audio_match.group(1).strip() if audio_match else ""

        return cls(text=text, visual=video, audio=audio)

class IronyClassifier:
    def __init__(self, model_api):
        self.model_api = model_api

    def predict_irony(self, data: ModalityData) -> str:
        instruction = """
        You are an expert in emotional intelligence, responsible for classifying sentiment into one of the following categories: anger, sadness, joy, neutral, fear, surprise, or disgust. Use the text modality as the primary basis for interpretation, while integrating text clues to adjust emotional classifications. In cases where there is a conflict between modalities, rely on the text to help distinguish subtle emotional differences. Respond with a single word corresponding to the detected emotion: anger, sadness, joy, neutral, fear, surprise, or disgust.
        """
        input_text = f"Video: {data.visual}; Text: {data.text}; Audio: {data.audio}"
        return self.model_api.get_response(instruction, input_text).strip().lower()

def remove_modalities(data: ModalityData, remove: List[str]) -> ModalityData:
    return ModalityData(
        text="" if "T" in remove else data.text,
        visual="" if "V" in remove else data.visual,
        audio="" if "A" in remove else data.audio
    )

def calculate_metrics(predictions, actual):
    tp = sum((p == "true" and a == "true") for p, a in zip(predictions, actual))
    fp = sum((p == "true" and a == "false") for p, a in zip(predictions, actual))
    fn = sum((p == "false" and a == "true") for p, a in zip(predictions, actual))
    tn = sum((p == "false" and a == "false") for p, a in zip(predictions, actual))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'support': len(predictions)
    }

def run_ablation_experiment(api, df, remove_modalities_list):
    classifier = IronyClassifier(api)
    results = []

    print("\nStarting ablation experiment...")
    print("-" * 50)

    for remove in remove_modalities_list:
        modality_name = "-".join(remove) if remove else "All"
        print(f"\nTesting modality combination: {modality_name}")
        
        predictions = []
        actual_values = []
        records = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {modality_name}"):
            modality_data = ModalityData.from_input_string(row['input'])
            ablated_data = remove_modalities(modality_data, remove)

            prediction = classifier.predict_irony(ablated_data)
            correct_answer = row['output'].strip().lower()

            predictions.append(prediction)
            actual_values.append(correct_answer)

            records.append({
                'Index': idx,
                'Removed_Modalities': modality_name,
                'Prediction': prediction,
                'Actual': correct_answer,
                'Correct': prediction == correct_answer
            })

        metrics = calculate_metrics(predictions, actual_values)
        
        results.append({
            'Removed_Modalities': modality_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1'],
            'Accuracy': metrics['accuracy'],
            'Support': metrics['support']
        })

        print(f"\nResults for {modality_name}:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Support: {metrics['support']}")
        print("-" * 30)

        df_results = pd.DataFrame(records)
        df_results.to_json(f'ablation_results/dcsa_ablation_{modality_name}.json', 
                          orient='records', lines=True)

    return pd.DataFrame(results)

if __name__ == "__main__":
    os.makedirs('ablation_results', exist_ok=True)
    
    api = DialogueModelAPI(model_path="your path")

    data_file = os.path.join('your path', 'test_dataset_extract.json')
    df = pd.read_json(data_file)

    remove_list = [
        ["V"],     
        ["T"],      
        ["A"],      
        ["V", "T"], 
        ["V", "A"], 
        ["T", "A"]  
    ]
    
    results_df = run_ablation_experiment(api, df, remove_list)
    
    results_df.to_json('ablation_results/dcsa_ablation_summary.json', 
                       orient='records', lines=True)
    
    print("\nFinal Summary:")
    print("=" * 80)
    print(results_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))
    print("=" * 80)
