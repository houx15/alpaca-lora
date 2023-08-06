import argparse
import os
import pandas as pd
import json

from utils.utils import sentence_cleaner


class DataGen(object):
    def __init__(self, topic, output_dir) -> None:
        self.topic = topic
        self.translator = self.load_result_translator()
        self.load_dataset()
        self.prompt = self.load_prompt_designer()

        # self.export_json_data(self.train_df, f"{output_dir}/train.json")
        # self.export_json_data(self.val_df, f"{output_dir}/val.json")
        # self.export_json_data(self.test_df, f"{output_dir}/test.json")
        self.export_json_data(self.total_df, f"{output_dir}/total.json")

    def load_dataset(self) -> None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dataset/{self.topic}')
        if self.topic == "china":
            data_dir += "/favorability"
        # self.train_df = pd.read_csv(os.path.join(data_dir, 'train-regression.csv'), header=0, usecols=['text', 'label'])
        # self.test_df = pd.read_csv(os.path.join(data_dir, 'test-regression.csv'), header=0, usecols=['text', 'label'])
        # self.val_df = pd.read_csv(os.path.join(data_dir, 'validate-regression.csv'), header=0, usecols=['text', 'label'])
        self.total_df = pd.read_csv(os.path.join(data_dir, 'total-regression.csv'), header=0, usecols=['text', 'label'])

        # self.df = pd.read_csv(os.path.join(data_dir, 'total-regression.csv'), header=0, usecols=['text', 'label'])

        # prompt based
        convert_label_dict = {}
        for k, v in self.translator['labels'].items():
            convert_label_dict[v] = k
        
        # self.train_df['label'] = self.train_df['label'].map(convert_label_dict)
        # self.val_df['label'] = self.val_df['label'].map(convert_label_dict)
        # self.test_df['label'] = self.test_df['label'].map(convert_label_dict)
        self.total_df['label'] = self.total_df['label'].map(convert_label_dict)
        # self.df['label'] = self.df['label'].map(convert_label_dict)

    def load_result_translator(self) -> dict:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/label_map.json'), 'r', encoding='utf8') as rfile:
            translator_dict = json.loads(rfile.read())
            return translator_dict[self.topic]
    
    def load_prompt_designer(self) -> str:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/prompt_design.json'), 'r', encoding='utf8') as rfile:
            prompt_dict = json.loads(rfile.read())
            return prompt_dict[self.topic]
    

    def export_json_data(self, df: pd.DataFrame, output: str) -> None:
        res = []
        for index, row in df.iterrows():
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

    for topic in ['gun', 'abortion', 'drug', 'climate', 'sexual']:
        generator = DataGen(topic=topic, output_dir=f'json/{topic}')
