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

### Instruction Text
We collect them from five data sources which are under these licenses: MIT, Apache-2.0, GPL-3.0, CC BY-NC 4.0.

### Input
We collect all sentimental adjectives from [sentiwordnet 3.0](https://github.com/aesuli/SentiWordNet?tab=readme-ov-file), such as "beautiful", "terrible", .etc., which is under [CC BY-SA 4.0 license](https://github.com/aesuli/SentiWordNet?tab=readme-ov-file).

### Output
All outputs are from the set of {"positive", "negative", "neutral"}

