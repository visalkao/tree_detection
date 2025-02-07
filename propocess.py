# preprocessing.py
import sys
from datasets import load_dataset
from PIL import Image
import numpy as np

def process_example(example):
    try:
        # Ensure image is in correct mode and size
        img = example['image'].convert('L').resize((256, 256), Image.BILINEAR)
        lbl = example['annotation'].convert('L').resize((256, 256), Image.NEAREST)
        
        return {
            'image': np.array(img, dtype=np.uint8),
            'label': np.array(lbl, dtype=np.uint8)
        }
    except Exception as e:
        print(f"Skipping corrupted example: {str(e)}", file=sys.stderr)
        return None

if __name__ == '__main__':
    ds = load_dataset("restor/tcd")
    processed = ds.map(
        process_example,
        batched=False,
        num_proc=4,
        remove_columns=ds['train'].column_names,
        load_from_cache_file=False
    ).filter(lambda x: x is not None)
    
    processed.save_to_disk("./processed_tcd_final")