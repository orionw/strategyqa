import os
import json
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

DATA_FOLDER = "../data/strategyqa/"

def make_answer_list(data_file: str, paragraph_file: str, out_file_path: str, use_cuda: bool = False):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad", truncation=True)
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    if use_cuda:
        model = model.cuda()

    output_file = open(os.path.join(DATA_FOLDER, out_file_path), "w")

    print("Loading data...")
    with open(os.path.join(DATA_FOLDER, data_file), "r") as fin:
        data = json.load(fin)
    
    with open(os.path.join(DATA_FOLDER, paragraph_file), "r") as fin:
        para = json.load(fin)

    print("Starting predictions...")
    all_instances_with_values = []
    for instance in tqdm.tqdm(data):
        all_answers = {"qid": instance["qid"], "decomposition": []}
        for idx, decomp in enumerate(instance["decomposition"]):
            decomp_answers = []
            decomp_para = []
            for annote in instance["evidence"]: # each annotator
                cur_titles: list = annote[idx] # first item contains wiki, 2nd contains others
                if "no_evidence" in cur_titles or "operation" in cur_titles:
                    decomp_answers.append("")
                    decomp_para.append("")
                else:
                    for title_groups in cur_titles: # should be only 1
                        assert len(cur_titles) == 1, cur_titles
                        for wiki_page in title_groups:
                            cur_para = para[wiki_page]
                            inputs = tokenizer(decomp, cur_para["content"], return_tensors='pt', truncation=True)
                            
                            if use_cuda:
                                inputs["input_ids"] = inputs["input_ids"].cuda()
                                inputs["attention_mask"] = inputs["attention_mask"].cuda()

                            start_logits, end_logits = model(**inputs)
                            answer_start = torch.argmax(start_logits)
                            answer_end = torch.argmax(end_logits) + 1
                            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
                            if answer not in ["[CLS]", "[PAD]"]:
                                decomp_answers.append(answer)
                                decomp_para.append(cur_para["content"])
            
            all_answers["decomposition"].append({"answers": decomp_answers, "contexts": decomp_para})
        output_file.write(json.dumps(all_answers)+ "\n")



if __name__ == "__main__":
    # make_answer_list("train.json", "strategyqa_train_paragraphs.json", "train_additional.jsonl", use_cuda=True)
    # before doing this line for dev, run `get_dev_contexts.py`
    make_answer_list("dev_ext.json", "strategyqa_dev_paragraphs.json", "dev_additional.jsonl", use_cuda=True)