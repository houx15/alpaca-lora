import argparse
import os
import pandas as pd
import json

from utils.utils import sentence_cleaner


class DataGen(object):

    def __init__(self, topic, task_type, output_dir) -> None:
        self.topic = topic
        self.task_type = task_type
        self.load_prompt_and_translator()
        self.load_dataset()

        self.export_json_data(self.train_df, f"{output_dir}/train-{task_type}.json")
        self.export_json_data(self.val_df, f"{output_dir}/val-{task_type}.json")
        self.export_json_data(self.test_df, f"{output_dir}/test-{task_type}.json")
        self.export_json_data(self.total_df, f"{output_dir}/total-{task_type}.json")
    
    def load_prompt_and_translator(self):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/prompt_design.json'), 'r', encoding='utf8') as rfile:
            prompt_dict = json.loads(rfile.read())
            self.full_name = prompt_dict["full_name"][self.topic]
    
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/label_map.json"), "r", encoding="utf8") as rfile:
            self.translator = json.loads(rfile.read())[self.topic]
            label_mappings = self.translator["labels"]
            labels_with_relevance = []
            labels_without_relevance = []
            for label_mapping in label_mappings.items():
                if label_mapping[1] != -9:
                    labels_with_relevance.append(label_mapping[0])
                    labels_without_relevance.append(label_mapping[0])
                else:
                    labels_with_relevance.append(label_mapping[0])
        if self.task_type == "binary":
            self.prompt = prompt_dict["binary"].format(full_name=self.full_name)
        elif self.task_type == "regression":
            self.prompt = prompt_dict["regression"].format(full_name=self.full_name, labels=', '.join(labels_without_relevance))
        elif self.task_type == "regression_with_relevance":
            self.prompt = prompt_dict["regression"].format(full_name=self.full_name, labels=', '.join(labels_without_relevance))

    def load_dataset(self) -> None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dataset/{self.topic}')
        if self.topic == "china":
            data_dir += "/favorability"
        self.train_df = pd.read_csv(os.path.join(data_dir, f'train-{self.task_type}.csv'), header=0, usecols=['text', 'label'])
        self.test_df = pd.read_csv(os.path.join(data_dir, f'test-{self.task_type}.csv'), header=0, usecols=['text', 'label'])
        self.val_df = pd.read_csv(os.path.join(data_dir, f'validate-{self.task_type}.csv'), header=0, usecols=['text', 'label'])
        self.total_df = pd.read_csv(os.path.join(data_dir, f'total-{self.task_type}.csv'), header=0, usecols=['text', 'label'])
        
        # prompt based
        convert_label_dict = {}
        if self.task_type == "regression" or self.task_type == "regression_with_relevance":
            for k, v in self.translator['labels'].items():
                convert_label_dict[v] = k
        elif self.task_type == "binary":
            convert_label_dict[0] = f"irrelevant to {self.full_name}"
            convert_label_dict[1] = f"relevant to {self.full_name}"
        
        self.train_df['label'] = self.train_df['label'].map(convert_label_dict)
        self.val_df['label'] = self.val_df['label'].map(convert_label_dict)
        self.test_df['label'] = self.test_df['label'].map(convert_label_dict)
        self.total_df['label'] = self.total_df['label'].map(convert_label_dict)
        # self.df['label'] = self.df['label'].map(convert_label_dict)

    def export_json_data(self, df: pd.DataFrame, output: str) -> None:
        res = []
        for _, row in df.iterrows():
            res.append({
                "instruction": self.prompt,
                "input": sentence_cleaner(row["text"]),
                "output": row["label"]
            })
        
        with open(output, "w", encoding="utf8") as wfile:
            json.dump(res, wfile)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Prediction/Evaluation.')
    
    # parser.add_argument('topic', type=str, choices=['gun', 'china', 'abortion', 'drug', 'climate', 'sexual'])
    # parser.add_argument('--output_dir', type=str, default='./data')

    # config = parser.parse_args()

    # evaluator = DataGen(topic=config.topic, output_dir=config.output_dir)

    for topic in ['gun', 'abortion', 'drug', 'climate', 'sexual', "china"]:
        for task_type in ["binary", "regression", "regression_with_relevance"]:
            output_dir = f"json/{topic}"
            if topic == "china":
                output_dir += "/favoribility"
            generator = DataGen(topic=topic, task_type=task_type, output_dir=output_dir)
