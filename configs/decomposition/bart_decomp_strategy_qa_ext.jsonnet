local config = import "base.jsonnet";

config {
  dataset_reader_type:: "strategy_decomposition_reader_extended",
  "train_data_path": "data/strategyqa/train.json",
  "validation_data_path": "data/strategyqa/dev.json",
  "train_additional_data_path": "data/strategy_qa/train_additional.jsonl",
  "validation_additional_data_path": "data/strategy_qa/dev_additional.jsonl",
}
