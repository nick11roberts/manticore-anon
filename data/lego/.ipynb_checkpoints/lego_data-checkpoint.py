import torch
import torch.nn as nn
import numpy as np
import os
import math

all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def generate_data(tokenizer, n_var, batch_size=100, random_order=True, append_vars_sep_token=None):
    
    batch = []
    labels = []
    clause_order = []
    for _ in range(batch_size):
        values = np.random.randint(0, 2, (n_var,))
        var_idx = tuple(np.random.permutation(len(all_vars)))
        vars = [all_vars[i] for i in var_idx]

        # generate first sentence
        clauses = []
        clauses.append('%s = val %d , ' % (vars[0], values[0]))
        # clauses.append(str(values[0])+' ')

        for i in range(1, n_var):
            modifier = 'val' if values[i] == values[i-1] else 'not'
            clauses.append(' %s = %s %s , ' % (vars[i], modifier, vars[i-1]))
            # clauses.append(' 1 ' if modifier=='val' else ' 0 ')
            

        sent = ''
        label = []
        
        clause_idx = tuple(np.random.permutation(n_var)) if random_order else tuple(range(n_var))
        sent += ''.join([clauses[idx] for idx in clause_idx])
        sent = " " + " ".join(sent.split()) # EDIT; sanitize out the multiple spaces 
        if append_vars_sep_token is not None:
            sent = " ".join([sent, append_vars_sep_token, *(vars[:n_var])])
        label += [values[idx] for idx in clause_idx]
        
        
        order = torch.zeros(1, n_var, n_var)
        for i in range(n_var):
            order[0, i, clause_idx[i]] = 1
            
        batch.append(tokenizer(sent, return_tensors='pt')['input_ids'])
        # print(tokenizer(sent, return_tensors='pt')['input_ids'].shape, sent)
        # print(tokenizer(sent, return_tensors='pt')['input_ids'])
        labels.append(values)
        clause_order.append(order)
    return torch.cat(batch), torch.LongTensor(labels), torch.cat(clause_order)

class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, lego_reg_dataset):
        self.data = lego_reg_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        x,y,ord=self.data.__getitem__(i)
        return {"input_ids":x, "labels":y, "order":ord}
        
def lego_dataset(tokenizer, n_var, n_samples, batch_size=100, random_order=True, append_vars_sep_token=None):
    # note:  batch size is not a batch size during training, just how many get generated at a time
    train_data = []
    train_labels = []
    train_order = []

    for i in range(n_samples//batch_size):
        batch, labels, order = generate_data(tokenizer, n_var, batch_size, random_order, append_vars_sep_token)
        train_data.append(batch)
        train_labels.append(labels)
        train_order.append(order)

    x_train = torch.cat(train_data)
    y_train = torch.cat(train_labels)
    order_train = torch.cat(train_order)
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train, order_train)
    trainset = LegoDataset(trainset)
    return trainset


def make_lego_datasets(tokenizer, n_var, n_train, n_test, batch_size=100, random_order=True, append_vars_sep_token=None):
    
    train_data = []
    train_labels = []
    train_order = []

    for i in range(n_train//batch_size):
        batch, labels, order = generate_data(tokenizer, n_var, batch_size, random_order, append_vars_sep_token)
        train_data.append(batch)
        train_labels.append(labels)
        train_order.append(order)

    x_train = torch.cat(train_data)
    y_train = torch.cat(train_labels)
    order_train = torch.cat(train_order)
    
    trainset = torch.utils.data.TensorDataset(x_train, y_train, order_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    test_data = []
    test_labels = []
    test_order = []
    for i in range(n_test//batch_size):
        batch, labels, order = generate_data(tokenizer, n_var, batch_size, random_order, append_vars_sep_token)
        test_data.append(batch)
        test_labels.append(labels)
        test_order.append(order)

    x_test = torch.cat(test_data)
    y_test = torch.cat(test_labels)
    order_test = torch.cat(test_order)

    testset = torch.utils.data.TensorDataset(x_test, y_test, order_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    return trainloader, testloader

