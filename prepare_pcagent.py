import os
from tqdm import tqdm
import requests
import base64
import json
import string
import pandas as pd
from vlmeval.smp.vlm import encode_image_file_to_base64

with open('/root/VLMEvalKit/Agent_Eval/benchmark.json', 'r') as file:
    benchmark_data = file.read()
benchmark_data = json.loads(benchmark_data)
benchmark_data = [
    {key.replace('answer_', ''): value for key, value in item.items()}
    for item in benchmark_data
]
benchmark_data = [
    {key.replace('image_path', 'image'): value for key, value in item.items()}
    for item in benchmark_data
]
benchmark_data = [
    {key: value for key, value in item.items() if key != 'hint'}
    for item in benchmark_data
]
benchmark_data = [
    {**item, 'image': encode_image_file_to_base64("/root/VLMEvalKit/Agent_Eval/" + item['image'])} if 'image' in item else item
    for item in tqdm(benchmark_data, desc="Encoding images")
]

benchmark_df = pd.DataFrame(benchmark_data)
benchmark_df.to_csv('/root/VLMEvalKit/Agent_Eval/benchmark.tsv', sep='\t', index_label='index')
