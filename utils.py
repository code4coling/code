#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch, random, os, json
import numpy as np

def read(path):
    data=[]
    with open(path, "r") as f:
        for line in f.readlines():
            temp = line.strip()
            if temp!="":
                data.append(temp)
    return data

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #if you are using multi-GPU
    np.random.seed(seed)  #numpy module
    random.seed(seed)  #python random modul
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)