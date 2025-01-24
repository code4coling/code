# Sentimental Adjective-based Instruction Tuning (SABIT) 
***The data we released in this reposity will be used for research purpose only. Our dataset is under CC BY-NC 4.0 license***

## SABIT data

The repository contains the constructed instruction tuple (Instruction text, Input, Output) using the method proposed in section 2 of the paper. An example is shown here: 
```json
{
    "instruction text": "In this task, you are asked to label the sentiment of a given sentence. The sentiment can be positive, negative or neutral.",
    "input": "excellent",
    "output": "positive"
},
```
Please download `data_240_sentiment_adj_with_negation.json` where there are 192 instances for training and 48 instances for development.

- Instruction Text
We collect them from five data sources which are under these licenses: MIT, Apache-2.0, GPL-3.0, CC BY-NC 4.0.

- Input
We collect all sentimental adjectives from [sentiwordnet 3.0](https://github.com/aesuli/SentiWordNet?tab=readme-ov-file), such as "beautiful", "terrible", .etc., which is under [CC BY-SA 4.0 license](https://github.com/aesuli/SentiWordNet?tab=readme-ov-file).

- Output
All outputs are from the set of {"positive", "negative", "neutral"}

## SABIT Training

## Prerequisite
Install the necessary packages to run training scripts. Note that instruction tuning requires the `peft` package to enable Low-Rank Adaptation (LoRA) and at least one A100 GPU is needed. 

```python 
pip install -r requirements.txt
```

## Training 
```shell
CUDA_VISIBLE_DEVICES=0 python train.py 
                --model_name_or_path   # llama2 base model or falcon model
                --lora_rank            # for rank in lora
                --lora_alpha           # for alpha in lora
                --lora_dropout         # for dropout in lora
                --seed                 # random seed   
                --lr                   # learning rate
                --out_dir              # output directory
                --nb_epoch             # number of epochs
                --gradient_accumulation_steps # Gradient accumulation steps
                --per_device_train_batch_size # Batch size per device
                --logging_steps        # logging steps in trainer
                --save_steps           # checkpoint save steps in trainer
                --eval_steps           # evaluation steps in trainer
                
```
The LoRA model will be saved in the output directory. Note: For Llama2 and Falcon models, different parameters are updated during training using the LoRA (Low-Rank Adaptation) approach.


