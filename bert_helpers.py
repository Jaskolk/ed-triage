
import torch
import random
import time
import math

import pandas as pd
import numpy as np
import collections
from collections import Counter

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import f1_score

from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import Optimizer

"""### define helper functions"""

def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

###

def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return sklearn.metrics.accuracy_score(labels_flat, preds_flat)

###

def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds==label])}/{len(y_true)}\n')

###

def evaluate(dataloader_val):

    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total/len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

###

def train_model(model, dataloader_train, dataloader_valid, save_name, optim = 'default', TPU = False, lr = 1e-5, eps = 1e-8, epochs = 3, weights = None):

  model.to(device)

  if weights:
    weight = torch.tensor([float(i) for i in weights.values()]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight = weight)
    print ('using weighted cross entropy loss')
    print (loss_fn)

  if optim == 'default':
    print ('optimizing with AdamW')
    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
  else:
    print ('optimizing with ', str(optim))
    optimizer = optim

  scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)

  for epoch in tqdm(range(1, epochs+1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        outputs = model(**inputs)

        if weights:
          loss = loss_fn(outputs[1], batch[2])
        else:
          loss = outputs[0]

        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if TPU:
          xm.optimizer_step(optimizer, barrier=True)
        else:
          optimizer.step()

        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})


    torch.save(model.state_dict(), f'/content/{save_name}_BERT_epoch_{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    val_acc = accuracy_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')
    tqdm.write(f'Validation accuracy: {val_acc}')

###

def get_metrics(dataloader): #(predictions, true_vals, dataloader):

  _, predictions, true_vals = evaluate(dataloader)

  preds = [np.argmax(pred) for pred in predictions]
  preds_flat = np.argmax(preds).flatten()
  true_vals = true_vals.flatten()

  f1_w = sklearn.metrics.f1_score(true_vals, preds, average='weighted')
  f1 = sklearn.metrics.f1_score(true_vals, preds, average=None)
  acc = sklearn.metrics.accuracy_score(true_vals, preds)
  prec = sklearn.metrics.precision_score(true_vals,preds, average=None)
  rec = sklearn.metrics.recall_score(true_vals,preds, average=None)
  auroc = sklearn.metrics.roc_auc_score(true_vals,predictions[:,1], average=None)
  confusion = sklearn.metrics.confusion_matrix(true_vals, preds)

#labels flipped for some reason so I had to chance the confusion interpretation
#tp, fn, fp, tn = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
  tn, fn, fp, tp = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]


  sens = tp/(tp + fn)
  spec = tn/(tn + fp)
  ppv = tp/(tp + fp)
  npv = tn/(tn + fn)

  print ('Metrics Report:')
  print ('---------------')
  print ('weighted f1: ', f1_w)
  print ('AUROC:       ',auroc)
  print ('accuracy:    ', acc)
  print ('precision:   ', prec)
  print ('recall:      ', rec)
  print ('sensitivity: ', sens)
  print ('specificity: ', spec)
  print ('PPV:         ', ppv)
  print ('NPV:         ', npv)
  print ()
  print ('confusion matrix')
  print (confusion)

  results_df.loc[len(results_df)] = [desc,num_samples, weights, f1_w, acc, auroc, ppv, sens, batch_size]

###

def encode_data(text_field):
  encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'][text_field].values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
    )

  encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'][text_field].values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='pt'
    )


  input_ids_train = encoded_data_train['input_ids']
  attention_masks_train = encoded_data_train['attention_mask']
  labels_train = torch.tensor(df[df.data_type=='train'].label.values)

  input_ids_val = encoded_data_val['input_ids']
  attention_masks_val = encoded_data_val['attention_mask']
  labels_val = torch.tensor(df[df.data_type=='val'].label.values)

  return input_ids_train, attention_masks_train, labels_train, input_ids_val, attention_masks_val, labels_val

###

def create_dataloaders(batch_size = 32):
  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
  dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

  dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(dataset_train),
                              batch_size=batch_size)

  dataloader_validation = DataLoader(dataset_val,
                                   sampler=SequentialSampler(dataset_val),
                                   batch_size=batch_size)

  return dataloader_train, dataloader_validation

###

class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, adam=False):   #I changed weight decay from original which was 0
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Paper v3 does not use debiasing.
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr']  * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(group['weight_decay'], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio, adam_step)

        return loss
