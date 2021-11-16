import os
import sys
import json
import torch
import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from src.data.dataset_readers.utils.elasticsearch_utils import (
    get_elasticsearch_paragraph,
    get_elasticsearch_results,
    concatenate_paragraphs,
    clean_query,
)

DATA_FOLDER = "../data/strategyqa/"
QUERIES_CACHE_PATH = "../data/strategyqa/queries_cache.json"

def gather_contexts(data_file: str, out_file_path: str):
    print("Loading data...")
    with open(os.path.join(DATA_FOLDER, data_file), "r") as fin:
        data = json.load(fin)

    if os.path.exists(QUERIES_CACHE_PATH):
        with open(QUERIES_CACHE_PATH, "r", encoding="utf8") as f:
            _queries_cache = json.load(f)

    print("Starting predictions...")
    all_paras = {}
    data_aug = []
    # Needs to gather both new evidence contexts and update them in the
    # main file so that we can use the `find_answers_to_decompositions.py` file
    missing = 0
    for instance in tqdm.tqdm(data):
        new_evidence = []
        for idx, decomp in enumerate(instance["decomposition"]):
            query = clean_query(decomp)
            results = get_elasticsearch_results(_queries_cache, query)
            if results is None:
                missing += 1
                print(missing)
                results = []
                evidence_ids = [[]]
            else:
                evidence_ids = [[item["evidence_id"] for item in results]]
                
            new_evidence.append(evidence_ids)
            for result in results:
                result_copy = result.copy()
                del result_copy["evidence_id"]
                all_paras[result["evidence_id"]] = result_copy

        new_instance = instance.copy()
        new_instance["evidence"] = [new_evidence]
        data_aug.append(new_instance)
    
    with open(os.path.join(DATA_FOLDER, out_file_path), "w") as fout:
        json.dump(all_paras, fout, indent=4)

    with open(os.path.join(DATA_FOLDER, data_file.replace(".json", "_ext.json")), "w") as fout:
        json.dump(data_aug, fout, indent=4)
            

if __name__ == "__main__":
    gather_contexts("dev.json", "strategyqa_dev_paragraphs.json")
