---
tags:
- trl
---

# PRM800K Dataset

## Summary

The PRM800K dataset is a processed version of [OpenAI's PRM800K](https://github.com/openai/prm800k), designed to train models using the [TRL library](https://github.com/huggingface/trl) for stepwise supervision tasks. It contains 800,000 step-level correctness labels for model-generated solutions to problems from the MATH dataset. This dataset enables models to learn and verify each step of a solution, enhancing their reasoning capabilities.

## Data Structure

- **Format**: [Standard](https://huggingface.co/docs/trl/main/dataset_formats#standard)
- **Type**: [Stepwise supervision](https://huggingface.co/docs/trl/main/dataset_formats#stepwise-supervision)

Columns:
- `"pompt"`: The problem statement.
- `"completions"`: A list of reasoning steps generated to solve the problem.
- `"labels"`: A list of booleans or floats indicating the correctness of each corresponding reasoning step.

This structure allows models to learn the correctness of each step in a solution, facilitating improved reasoning and problem-solving abilities.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/prm800k.py).
