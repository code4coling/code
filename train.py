#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, torch
import utils
import SentimentDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, default_data_collator
from contextlib import nullcontext
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_int8_training
)
profiler = nullcontext()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='llama2-7b', help="Path to the pre-trained model")
    parser.add_argument('--data_dir', type=str, default='./data/train_and_val_240_our_sentiwordnet_word_with_negation.json', help="Path to dataset")
    parser.add_argument('--lora_rank', type=int, default=8, help="Rank for LoRA")
    parser.add_argument('--lora_alpha', type=int, default=32, help="Alpha for LoRA")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="Dropout for LoRA")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--nb_epoch', type=int, default=3, help="Number of epochs")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--logging_steps', type=int, default=1, help="logging steps for trainer")
    parser.add_argument('--save_steps', type=int, default=10, help="save steps for trainer")
    parser.add_argument('--eval_steps', type=int, default=10, help="eval steps for trainer")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device")
    parser.add_argument('--gradient_checkpointing', type=bool, default=False, help="Enable gradient checkpointing")
    return parser.parse_args()


def create_peft_config(model, args):
    """Create and return a PEFT configuration."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"] if 'llama' in args.model_name_or_path else ["query_key_value"]
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, lora_config


if __name__=='__main__':
    args = parse_arguments()
    utils.set_seed(args.seed)
    
    if 'llama' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", truncation_side="left", model_max_length=256)
        #model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=False, torch_dtype=torch.float32)
        tokenizer.pad_token = tokenizer.bos_token
        model.config.pad_token_id = model.config.bos_token_id   
    elif 'falcon' in args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left", truncation_side="left", trust_remote_code=True,return_token_type_ids=False, model_max_length=256)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    else:
        raise AssertionError('Unsupported model (should be either "llama2" or "falcon"')
    
    train_dataset = SentimentDataset.SentimentInstructionDataset(args.data_dir, tokenizer, split="train")
    val_dataset = SentimentDataset.SentimentInstructionDataset(args.data_dir, tokenizer, split="val")
    
    model, lora_config = create_peft_config(model, args)
    output_dir = f"./lora_model/lora_model_{args.model_name_or_path.split('/')[-1]}_sentiwordnet_word_with_negation_epoch{args.nb_epoch}_lr{args.lr}_lora-rank{args.lora_rank}_lora-alpha{args.lora_alpha}_loss-mask_eval-loss"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        bf16=True,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        evaluation_strategy='steps',
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        **{
            'learning_rate': args.lr,
            'num_train_epochs': args.nb_epoch,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'per_device_train_batch_size': args.per_device_train_batch_size,
            'gradient_checkpointing': args.gradient_checkpointing,
        }
    )
    
    with profiler:
        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            callbacks= [],
        )
        # Start training
        trainer.train()
    
    
    model.save_pretrained(training_args.output_dir)
    print(f"Model training complete. Model saved at {training_args.output_dir}")
