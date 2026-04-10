---
language:
- en
license: odc-by
size_categories:
- 1K<n<10K
task_categories:
- question-answering
pretty_name: RM Bench
dataset_info:
  features:
  - name: prompt
    dtype: string
  - name: chosen
    dtype: string
  - name: chosen_model
    dtype: string
  - name: rejected
    dtype: string
  - name: rejected_model
    dtype: string
  - name: subset
    dtype: string
  - name: id
    dtype: int64
  splits:
  - name: raw
    num_bytes: 10837043
    num_examples: 5123
  - name: filtered
    num_bytes: 4849207
    num_examples: 2985
  download_size: 7943951
  dataset_size: 15686250
configs:
- config_name: default
  data_files:
  - split: raw
    path: data/raw-*
  - split: filtered
    path: data/filtered-*
---

<img src="https://huggingface.co/spaces/allenai/reward-bench/resolve/main/src/logo.png" alt="RewardBench Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

[Code](https://github.com/allenai/reward-bench) | [Leaderboard](https://huggingface.co/spaces/allenai/reward-bench) | [Prior Preference Sets](https://huggingface.co/datasets/allenai/pref-test-sets) | [Results](https://huggingface.co/datasets/allenai/reward-bench-results) | [Paper](https://arxiv.org/abs/2403.13787)

# Reward Bench Evaluation Dataset Card

The RewardBench evaluation dataset evaluates capabilities of reward models over the following categories:
1. **Chat**: Includes the easy chat subsets (alpacaeval-easy, alpacaeval-length, alpacaeval-hard, mt-bench-easy, mt-bench-medium)
2. **Chat Hard**: Includes the hard chat subsets (mt-bench-hard, llmbar-natural, llmbar-adver-neighbor, llmbar-adver-GPTInst, llmbar-adver-GPTOut, llmbar-adver-manual)
3. **Safety**: Includes the safety subsets (refusals-dangerous, refusals-offensive, xstest-should-refuse, xstest-should-respond, do not answer)
4. **Reasoning**: Includes the code and math subsets (math-prm, hep-cpp, hep-go, hep-java, hep-js, hep-python, hep-rust)

The RewardBench leaderboard averages over these subsets and a final category from [prior preference data test sets](https://huggingface.co/datasets/allenai/preference-test-sets) including Anthropic Helpful, Anthropic HHH in BIG-Bench, Stanford Human Preferences (SHP), and OpenAI's Learning to Summarize data.

The scoring for RewardBench compares the score of a prompt-chosen pair to a prompt-rejected pair. 
Success is when the chosen score is higher than rejected.

<img src="https://huggingface.co/datasets/allenai/blog-images/resolve/main/reward-bench/scoring.png" alt="RewardBench Scoring" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>

In order to create a representative, single evaluation score, we perform a limited mixture of averaging across results.
For all the subsets detailed below except for Reasoning, we perform per-prompt weighted averaging across all the prompts in the subset to get the section score.
For example, in Chat we take a weighted average of the AlpacaEval and MT Bench sets based on the number of prompts.
For Reasoning, we increase the weight of the PRM-Math subset so code and math abilities are weighed equally in the final number, rather than increasing the relevance of code.
Once all subsets weighted averages are achieved, the final RewardBench score is the average across the subset scores (including Prior Sets).

## Dataset Details

In order to maintain all the relevant data, the samples in the dataset will have the following items. 
Note, the dataset is single-turn:
* `prompt` (`str`): the instruction given in the various test sets.
* `chosen` (`str`): the response from the better model or the better rated prompt.
* `chosen_model` (`str`): where applicable
* `rejected` (`str`): the response with the lower score or from word model.
* `rejected_model` (`str`): where applicable
* `subset` (`str`): the subset (e.g. alpacaeval-easy) of the associated prompt as the dataset is all in one split.
* `id` (`int`): an incremented id for every prompt in the benchmark.

To select a specific subset use HuggingFace Datasets `.filter` functionality.
```
dataset = dataset.filter(lambda ex: ex["subset"] == "alpacaeval-easy")
```

This can easily be converted to the standard chosen/rejected list of messages format (see [UltraFeedback for an example](https://huggingface.co/datasets/allenai/ultrafeedback_binarized_cleaned)), for example with our data loading utilities on [GitHub](https://github.com/allenai/reward-bench/blob/8eadb09397d58f1930d4f77938e618b9f9b8aeb3/rewardbench/utils.py#L330).


### Subset Summary


Total number of the prompts is: 2985. 

| Subset             | Num. Samples (Pre-filtering, post-filtering) | Description |
| :---------- | :-----: | :---------: |
| alpacaeval-easy    | 805, 100          | Great model vs poor model; GPT4-Turbo 97.7% v. Alpaca 7b 26.46% (data [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/results))          |
| alpacaeval-length    | 805, 95          | Good model vs low model, similar length; Llama2chat 70B 92.66% vs Guanaco 13B 52.61% (data [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/results)) |
| alpacaeval-hard    | 805, 95          | Great model vs baseline model; Tulu 2 95.0% v. Davinici003 50.0% (data [here](https://github.com/tatsu-lab/alpaca_eval/tree/main/results))|
| mt-bench-easy      | 28, 28           | MT Bench 10s vs 1s (source [data](https://huggingface.co/spaces/lmsys/mt-bench/tree/main/data/mt_bench))            |
| mt-bench-medium    | 45, 40           | MT Bench 9s vs 2-5s (source [data](https://huggingface.co/spaces/lmsys/mt-bench/tree/main/data/mt_bench))           |
| mt-bench-hard      | 45, 37          | MT Bench 7-8 vs 5-6  (source [data](https://huggingface.co/spaces/lmsys/mt-bench/tree/main/data/mt_bench))          |
| refusals-dangerous | 505, 100          | Dangerous rejected response vs polite chosen refusal            |
| refusals-offensive | 704, 100          | Offensive rejected response vs polite chosen refusal             |
| llmbar-natural     | 100          |  Manually curated instruction pairs (See [paper](https://arxiv.org/abs/2310.07641)) |
| llmbar-adver-neighbor | 134          |  Adversarial instruction response vs. off-topic prompt response (See [paper](https://arxiv.org/abs/2310.07641))|
| llmbar-adver-GPTInst | 92          |   Adversarial instruction response vs. GPT4 generated off-topic prompt response (See [paper](https://arxiv.org/abs/2310.07641))|
| llmbar-adver-GPTOut |  47          |   Adversarial instruction response vs. unhelpful-prompted GPT4 responses (See [paper](https://arxiv.org/abs/2310.07641))|
| llmbar-adver-manual |  46          |   Challenge set manually designed chosen vs. rejected |
| xstest-should-refuse | 450, 250         | False response dataset (see [paper](https://arxiv.org/abs/2308.01263))        |
| xstest-should-respond | 450, 154         | False refusal dataset (see [paper](https://arxiv.org/abs/2308.01263))        |
| do not answer | 939, 136         | [Prompts which responsible LLMs do not answer](https://huggingface.co/datasets/LibrAI/do-not-answer): Refusals are chosen and responses are rejected         |
| hep-cpp | 164         | C++ working code vs. buggy code (See [dataset](https://huggingface.co/datasets/bigcode/humanevalpack) or [paper](https://arxiv.org/abs/2308.07124))        |
| hep-go | 164         |   Go working code vs. buggy code       |
| hep-java | 164         |  Java working code vs. buggy code        |
| hep-js | 164         |    Javascript working code vs. buggy code        |
| hep-python | 164         |  Python working code vs. buggy code         |
| hep-rust | 164         |   Rust working code vs. buggy code        |
| math-prm | 447         | Human references vs. model error (see [paper](https://github.com/openai/prm800k))        |

The length distribution of the subsets with a Llama tokenizer is shown below.

| subset                |   Chosen Mean Tokens |   Rejected Mean Tokens |   Chosen Max Tokens |   Rejected Max Tokens |   Chosen Min Tokens |   Rejected Min Tokens |   Chosen Mean Unique Tokens |   Rejected Mean Unique Tokens |   Chosen Max Unique Tokens |   Rejected Max Unique Tokens |   Chosen Min Unique Tokens |   Rejected Min Unique Tokens |
|-----------------------|----------------------|------------------------|---------------------|-----------------------|---------------------|-----------------------|-----------------------------|-------------------------------|----------------------------|------------------------------|----------------------------|------------------------------|
| alpacaeval-easy       |             591.26   |                167.33  |                1332 |                  1043 |                  40 |                    15 |                    252.91   |                       83.44   |                        630 |                          290 |                         33 |                           12 |
| alpacaeval-hard       |             411.684  |                136.926 |                1112 |                   711 |                  57 |                    12 |                    172.537  |                       70.9684 |                        359 |                          297 |                         45 |                            8 |
| alpacaeval-length     |             510.589  |                596.895 |                1604 |                  2242 |                  55 |                    52 |                    192.442  |                      188.547  |                        434 |                          664 |                         30 |                           38 |
| donotanswer           |             169.61   |                320.5   |                 745 |                   735 |                  20 |                    20 |                    103.743  |                      156.941  |                        358 |                          337 |                         18 |                           13 |
| hep-cpp               |             261.262  |                259.488 |                 833 |                   835 |                  53 |                    57 |                     99.8537 |                       99.372  |                        201 |                          201 |                         37 |                           40 |
| hep-go                |             266.22   |                264.598 |                 732 |                   720 |                  55 |                    57 |                     99.622  |                       99.189  |                        201 |                          201 |                         36 |                           37 |
| hep-java              |             263.14   |                260.939 |                 748 |                   733 |                  55 |                    54 |                    102.311  |                      101.927  |                        207 |                          206 |                         39 |                           41 |
| hep-js                |             251.165  |                249.695 |                 771 |                   774 |                  53 |                    52 |                     93.2744 |                       92.9268 |                        192 |                          192 |                         37 |                           40 |
| hep-python            |             211.988  |                211.146 |                 624 |                   612 |                  53 |                    49 |                     85.6463 |                       85.3049 |                        190 |                          190 |                         36 |                           35 |
| hep-rust              |             221.256  |                219.049 |                 988 |                   993 |                  46 |                    49 |                     95.1402 |                       94.8354 |                        192 |                          192 |                         36 |                           36 |
| llmbar-adver-GPTInst  |             170.109  |                377.359 |                 636 |                   959 |                  15 |                    15 |                     92.9457 |                      179.37   |                        287 |                          471 |                         12 |                           13 |
| llmbar-adver-GPTOut   |              96.4255 |                101     |                 393 |                   476 |                  18 |                    20 |                     60.0426 |                       55.0426 |                        241 |                          228 |                         13 |                           14 |
| llmbar-adver-manual   |             159.804  |                264.37  |                 607 |                   737 |                  23 |                    33 |                     91.9565 |                      140.13   |                        273 |                          385 |                         18 |                           24 |
| llmbar-adver-neighbor |              70.2239 |                172.507 |                 603 |                   865 |                   9 |                    13 |                     43.3134 |                       90.9328 |                        250 |                          324 |                          8 |                            9 |
| llmbar-natural        |             139.42   |                129.82  |                 907 |                   900 |                  17 |                    18 |                     74.99   |                       70.07   |                        354 |                          352 |                         14 |                           14 |
| math-prm              |             279.313  |                488.841 |                1608 |                  1165 |                  35 |                    77 |                     83.6264 |                      124.582  |                        237 |                          257 |                         23 |                           46 |
| mt-bench-easy         |             391.821  |                481.929 |                 778 |                  1126 |                 155 |                    31 |                    169.071  |                      121.321  |                        288 |                          434 |                         74 |                           19 |
| mt-bench-hard         |             287.784  |                301.649 |                 573 |                  1176 |                  68 |                    62 |                    133.622  |                      121.676  |                        261 |                          309 |                         50 |                           48 |
| mt-bench-med          |             351.375  |                466.025 |                 655 |                  1297 |                 145 |                    52 |                    159.9    |                      140.325  |                        285 |                          495 |                         82 |                           41 |
| refusals-dangerous    |             208.4    |                458.61  |                 380 |                   804 |                  87 |                   103 |                    128.53   |                      211      |                        200 |                          365 |                         71 |                           55 |
| refusals-offensive    |             139.82   |                298.63  |                 278 |                  1117 |                  75 |                    26 |                     95.98   |                      134.02   |                        170 |                          477 |                         60 |                           19 |
| xstest-should-refuse  |             129.227  |                217.019 |                 402 |                   549 |                  18 |                    15 |                     80.5519 |                      116.149  |                        194 |                          245 |                         16 |                           13 |
| xstest-should-respond |             188.708  |                107.356 |                 515 |                   465 |                  20 |                    16 |                    103.788  |                       67.328  |                        231 |                          202 |                         15 |                           16 |

### Filtering Summary

The RewardBench dataset is manually filtered from 5123 source prompts to manually verify the chosen-rejected ranking of prompts.
* The categories of AlpacaEval and MT Bench are manually filtered for every prompt.
* LLMBar, DoNotAnswer, HEP, and Math PRM all contained structured metadata for automatic filtering.
* XSTest is a hybrid of manual confirmation with metadata from the project.
* Refusals are automatically generated as a refusal or response (where refusal is preffered) with manual confirmation.

Substantial filtering details are available in the appendix of the papr.
If there are any bugs in the data, please reach out!


### License information

Licensing an aggregated dataset is a complex task. 
We release the RewardBench dataset under [ODC-BY](https://opendatacommons.org/licenses/by/) requiring the user to follow the licenses of the subsequent parts.
Licensing LLM datasets is an evolving topic. The licenses primarily apply to the prompts and the completions generated by models are often unlicensed.
The details for the datasets used in this work vary in the level of the detail on licenses and method of applying them.

| Dataset       | Variants                                                 | Data License                                              |
|---------------|----------------------------------------------------------|------------------------------------------------------|
| AlpacaEval    | {Easy, Length, Hard}                                        | [CC By NC 4.0](https://github.com/tatsu-lab/alpaca_farm/blob/main/DATA_LICENSE)  |
| MT Bench      | {Easy, Medium, Hard}                                        | [Apache 2.0](https://github.com/lm-sys/FastChat/blob/main/LICENSE)                                           |
| LLMBar        | {Natural, Neighbor, GPTInst, GPTOut, Manual} | [MIT License](https://github.com/princeton-nlp/LLMBar?tab=MIT-1-ov-file)                                          |
| Do Not Answer |                                                          | [CC BY NC SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)                                      |
| XSTest        | {Should Respond, Should Refuse}                          | [CC By 4.0](https://github.com/paul-rottger/exaggerated-safety?tab=CC-BY-4.0-1-ov-file)                                            |
| HumanEvalPack | {HEP CPP, Go, Javascript, Rust, Python, Rust}              | [MIT License](https://github.com/bigcode-project/octopack?tab=MIT-1-ov-file)                                          |
| PRM Math      |                                                          | [MIT License](https://github.com/openai/prm800k?tab=MIT-1-ov-file)                                          |

Within this dataset are prompts created by AI2 (the refusals data, released as MIT for now, see official release soon) with completions from API and open models. 
More details will come on this soon.

## Development 

### Requirements
Building the dataset requires `datasets`.
Maintaining the script and notebooks requites `notebook`.
```
pip install datasets notebook nbconvert
```
Convert with:
```
jupyter nbconvert --to script [YOUR_NOTEBOOK].ipynb
```
With no changes to the ipynb, the dataset can be re-built and pushed with the following (PLEASE BE CAREFUL):
```
python build_dataset.py
```

### Git LFS notes
If your uploads fail with:
```
Git LFS upload failed:  14% (1/7), 4.2 MB | 0 B/s                                                                                                                                                 
  (missing) data/train-00000-of-00001.parquet (425c88744455a9b0e7248cdd81fe4716085aae22849798f653f59fc878117a4d)
hint: Your push was rejected due to missing or corrupt local objects.
hint: You can disable this check with: `git config lfs.allowincompletepush true`
```
First fetch all lfs objects:
```
git lfs fetch --all origin main
```

### Filtering script (basic)
To filter data, run the following script:
```
python scripts/filter.py subset-name 0
```
with a subset from the dataset and a start index.

---

## Citation
```
@misc{RewardBench,
    title={RewardBench: Evaluating Reward Models for Language Modeling},
    author={Lambert, Nathan and Pyatkin, Valentina and Morrison, Jacob and Miranda, LJ and Lin, Bill Yuchen and Chandu, Khyathi and Dziri, Nouha and Kumar, Sachin and Zick, Tom and Choi, Yejin and Smith, Noah A. and Hajishirzi, Hannaneh},
    year={2024},
    howpublished={\url{https://huggingface.co/spaces/allenai/reward-bench}
}
```