import pandas as pd
import yaml
from pathlib import Path

def debug_data():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    fake_path = config["data"]["raw_fake_path"]
    true_path = config["data"]["raw_true_path"]
    
    print(f"Checking {fake_path}...")
    df_fake = pd.read_csv(fake_path)
    print(f"Fake rows: {len(df_fake)}")
    print(f"Fake columns: {df_fake.columns.tolist()}")
    print(f"Fake nulls in text: {df_fake['text'].isnull().sum()}")
    
    print(f"\nChecking {true_path}...")
    df_true = pd.read_csv(true_path)
    print(f"True rows: {len(df_true)}")
    print(f"True columns: {df_true.columns.tolist()}")
    print(f"True nulls in text: {df_true['text'].isnull().sum()}")
    
    df_fake["label"] = 1
    df_true["label"] = 0
    df = pd.concat([df_fake, df_true]).reset_index(drop=True)
    print(f"\nCombined rows: {len(df)}")
    print(f"Label counts:\n{df['label'].value_counts()}")
    
    initial_len = len(df)
    df = df.dropna(subset=["text"])
    print(f"\nAfter dropna: {len(df)}")
    
    df = df.drop_duplicates(subset=["text"])
    print(f"After drop_duplicates: {len(df)}")
    print(f"Final Label counts:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    debug_data()
