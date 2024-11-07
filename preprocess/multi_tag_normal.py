import json
import pandas as pd
import pyarrow.parquet as pq
import os
import re


def transform_keys(data):
    key_mapping_multi = {
        "input": "user",
        "question": "user",
        "output": "assistant",
        "answers": "assistant",
        "response": "assistant",
        "context": "context",
        "prefix": "context",
        "story": "context",
        "system": "context",
        "instruction": "instruction",
        "prompt": "instruction",
        "conversation": "messages",
        "conversations": "messages",
        "plain_text": "messages",
        "turn": "messages",
        "dialog": "messages",
        "history": "messages"
    }

    def transform_message_list(messages):
        if not messages:
            return messages
        elif isinstance(messages, str):
            return transform_message_string(messages)
        elif isinstance(messages[0], dict):
            for msg in messages:
                if 'from' in msg:
                    msg['role'] = msg.pop('from')
                if 'value' in msg:
                    msg['content'] = msg.pop('value')
                if 'speaker' in msg:
                    msg['role'] = msg.pop('speaker')
                if 'text' in msg:
                    msg['content'] = msg.pop('text')
            return messages
        elif isinstance(messages[0], list):
            transformed_messages = []
            for i, pair in enumerate(messages):
                if len(pair) == 2:
                    transformed_messages.append({"role": "user", "content": pair[0]})
                    transformed_messages.append({"role": "assistant", "content": pair[1]})
            return transformed_messages
        elif isinstance(messages[0], str):
            transformed_messages = []
            for i, msg in enumerate(messages):
                role = "user" if i % 2 == 0 else "assistant"
                transformed_messages.append({"role": role, "content": msg})
            return transformed_messages
        return messages

    def transform_message_string(message):
        pattern = re.compile(r'\[([a-zA-Z]+)\]: (.*?)(<eoa>|<eoh>)', re.DOTALL)
        matches = pattern.findall(message)
        transformed_messages = []
        role_mapping = {
            "Human": "user",
            "MOSS": "assistant"
        }
        for match in matches:
            role, content, _ = match
            if role in role_mapping:
                transformed_messages.append({"role": role_mapping[role], "content": content.strip()})
        return transformed_messages

    def transform(data, key_mapping):
        transformed_data = {}
        lower_key_mapping = {k.lower(): v for k, v in key_mapping.items()}
        for key, value in data.items():
            new_key = lower_key_mapping.get(key.lower(), key)
            if new_key == "messages" and isinstance(value, (list, str)):
                transformed_data[new_key] = transform_message_list(value)
            else:
                transformed_data[new_key] = value
        return transformed_data

    if isinstance(data, dict):
        if any(key.lower() in key_mapping_multi for key in data):
            return transform(data, key_mapping_multi)
    elif isinstance(data, list):
        return [transform(item, key_mapping_multi) for item in data if
                any(key.lower() in key_mapping_multi for key in item)]
    return data


def process_file(file_path, output_path):
    file_extension = file_path.split('.')[-1].lower()
    input_file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_path, f'{input_file_name}_processed.jsonl')

    if file_extension == 'json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        transformed_data = transform_keys(data)
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(transformed_data, list):
                for item in transformed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')

    elif file_extension == 'jsonl':
        with open(file_path, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                data = json.loads(line)
                transformed_data = transform_keys(data)
                f_out.write(json.dumps(transformed_data, ensure_ascii=False) + '\n')

    elif file_extension == 'parquet':
        df = pq.read_table(file_path).to_pandas()
        transformed_data = df.apply(lambda row: transform_keys(row.to_dict()), axis=1).tolist()
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in transformed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Transformed data saved to {output_file}")



