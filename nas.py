import torch
import torch.nn as nn
from transformers import Trainer

from functools import partial

def normalize(x, dim, min_v=1e-5):
    normed = x / x.sum(dim=dim, keepdim=True)
    normed = torch.clamp(normed, min=min_v)
    return normed

# Messy placeholder initialization
def model_kaiming_init(model):
    for _, param in model.named_parameters():
        if len(param.size())>=2:
            nn.init.kaiming_uniform_(param)
        else:
            nn.init.normal_(param,std=0.01)
'''Optimizer classes; some taken from dash/relax repos'''
class MixedOptimizer(torch.optim.Optimizer):
    def __init__(self, 
                 optimizers, 
                 alternating=True,
                 steps_per_opt:list=None, 
                ):
        '''
        Args:
            optimizers: list of objects that are subclasses of optim.Optimizer
            alternating: whether to alternate steps with different optimizers
            steps_per_opt: number of steps to spend on each specific optimizer; exactly analogous to steps_per_dataset in MixedDataset
            - In fact, these should run in synced parallel to each other; 1 to 1 correspondence of optimizer/dataset pairs
        '''

        self.optimizers = []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['method'] = type(optimizer)
                group['initial_lr'] = group.get('initial_lr', group['lr'])
            self.optimizers.append(optimizer)
        super(MixedOptimizer, self).__init__((g for o in self.optimizers for g in o.param_groups), {})
        self.alternating = alternating
        self.iteration = 0
        self.steps_per_opt = steps_per_opt if steps_per_opt is not None else [1]*len(optimizers)
        self.idx_list = sum( [ [ii]*steps for ii, steps in enumerate(self.steps_per_opt) ] , [])

    def step(self, *args, **kwargs):
        if self.alternating:
            idx = self.idx_list[ int(self.iteration % len(self.idx_list)) ]
            # print("OPT ", idx, " ", self.iteration)
            optimizer = self.optimizers[idx]
            try:
                optimizer.step(**kwargs)
            except TypeError:
                kwargs = {k:v for k,v in kwargs.items() if k=="closure"}
                optimizer.step(**kwargs)
        else:
            for optimizer in self.optimizers:
                try:
                    optimizer.step(**kwargs)
                except TypeError:
                    kwargs = {k:v for k,v in kwargs.items() if k=="closure"}
                    optimizer.step(**kwargs)
        self.iteration += 1
        
# Look at dash and relax/nas.py
class ExpGrad(torch.optim.Optimizer): # exponential update for GAEA
    def __init__(self, params, lr):
        params = list(params)
        for param in params:
            # if param.sum() - 1.0 > 0.01:
            #     param /= param.sum()
                # raise(ValueError("parameters must lie on the simplex"))
            if param.min()<0:
                param -= param.min() ## shift to all non-negative
            p_sum = param.sum()
            if p_sum==0.:
                param.data += 1.
            elif p_sum==1.:
                continue
            param.data = normalize(param.data, -1) ## simplex; sum to 1 
        super(ExpGrad, self).__init__(params, {'lr': lr})
    
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                p.data *= torch.exp( torch.clip(-lr*p.grad, -9, 9) )
                # p.data /= p.data.sum()
                p.data = normalize(p.data, -1)

def create_manticore_mixed_optimizer(manticore_model,
                                     weight_partial_opt,
                                     arch_lr,
                                     gaea=False,
                                     alternating=False,
                                     steps_per_opt:list=None):
    weight_list = []
    alphas_list = []
    for n,p in manticore_model.named_parameters():
        '''
        Alternatively, grab model.mixtures, model.gptneo, model.mamba?
        Although if there are any attribute naming inconsistencies in the different versions of manticore 1&2 that might be a headache
        Both use "alphas" for arch params
        '''
        if n.split(".")[-1] == "alphas":
            alphas_list.append(p)
        else:
            weight_list.append(p)
    print(alphas_list)
    weight_opt = weight_partial_opt(weight_list) ## default AdamW 
    if gaea:
        alphas_opt = ExpGrad(params=alphas_list, lr=arch_lr)
    else:
        alphas_opt = torch.optim.AdamW(params=alphas_list, lr=arch_lr)
    opt = MixedOptimizer([weight_opt, alphas_opt], alternating=alternating, steps_per_opt=steps_per_opt)
    return opt
'''
We need this custom Trainer since HP search requires a model_init not model, which means we can't pass our custom optimizer without the actual model already initialized
'''
class MixedOptTrainer(Trainer):
    def __init__(
        self,
        arch_lr,
        gaea=False,
        alternating=False,
        steps_per_opt=None,
        **orig_trainer_args,
    ):
        super().__init__(
            **orig_trainer_args
        )
        self.arch_lr = arch_lr
        self.alternating = alternating
        self.steps_per_opt = steps_per_opt
        self.gaea = gaea
        '''
        model = model,
        args = args,
        data_collator = data_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        model_init = model_init,
        compute_metrics = compute_metrics,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        '''
    def create_optimizer(self): ## overload func
        '''
        Lots of potential holes in this that won't work with things in the original Trainer func, such as sagemaker stuff for one
        IDK what else it'll break rn
        '''
        manticore_model = self.model
        wpo_kwargs = dict(
            lr= self.args.learning_rate,  
            betas= (self.args.adam_beta1, self.args.adam_beta2),
            weight_decay= self.args.weight_decay
        )
        wpo_kwargs = {k:v for k,v in wpo_kwargs.items() if v is not None}
        weight_partial_opt = partial(
            torch.optim.AdamW, 
            **wpo_kwargs,
        )
        
        self.optimizer = create_manticore_mixed_optimizer(
            manticore_model,
            weight_partial_opt,
            self.arch_lr,
            self.gaea,
            self.alternating,
            self.steps_per_opt
        )
        return self.optimizer
    

''''''
class MixedDataset(torch.utils.data.IterableDataset):
    def __init__(self, 
                 dataset_list:list, 
                 batch_size:int,
                 steps_per_dataset:list=None,
                ):
        '''
        dataset_list: list: 
        - list of datasets to cycle thru
        batch_size: int: 
        - the batch size we're going to use for following dataloading
        steps_per_dataset: list: 
        - how many consecutive steps to spend on each dataset before moving onto the next one. If not passed, 1 step each

        NOTES:
        - Currently only works for torch dataset classes, not HF datasets 
        '''
        # for dataset in dataset_list:
        #     assert isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, torch.utils.data.IterableDataset) , "Wrong dataset type somewhere in list; " + str(type(dataset))
        self.dataset_list = [dataset.__iter__() if (isinstance(dataset, torch.utils.data.IterableDataset) ) else dataset \
                                for dataset in dataset_list ] 
        # for dataset in self.dataset_list:
        #     print(type(dataset))
        self.batch_size = batch_size
        for s in steps_per_dataset:
            assert isinstance(s, int) , "Requires int number of steps for each dataset"
        self.steps_per_dataset = steps_per_dataset if steps_per_dataset is not None else [1]*len(dataset_list)
        
    def generate(self):
        '''
        Cycles thru dataset list; we want to return a batch from one, then move onto the next, repeat, etc. 
        - Requires we know the batch size ahead of time as is consistent with the batch size passed to the corresponding dataloader/HF trainer outside this dataset instance
        - Requires the batch to stay constant
        - 
        '''
        # i=0
        i=-self.batch_size 
        # In HF trainer the dataset apparent gets put 1 step ahead of the optimizer...?? 1 data batch gets loaded without an optimizer step, then all following batches have opt.step() afterward
        # Is it due to find_executable_batch_size??
        # https://github.com/huggingface/transformers/blob/835de4c8335f72a9c53178f54cc3b4c0688960ec/src/transformers/trainer.py#L1869 ??
        # If this is the case, then we need to make sure the given batch size can actually fit in memory, otherwise more samples will be iterated thru before opt.step()'s, and the batch_size in the Trainer will change to something different from the recorded batch size here.      
        # It seems the first batch here: https://github.com/huggingface/transformers/blob/835de4c8335f72a9c53178f54cc3b4c0688960ec/src/transformers/trainer.py#L2150
        #  ...doesn't step into the loop body
        
        n = len(self.dataset_list)
        intra_dataset_idxs = [0 if (hasattr(dataset, "__len__") ) else None for dataset in self.dataset_list]
        dataset_lens = [len(dataset) if (hasattr(dataset, "__len__") ) else None for dataset in self.dataset_list]

        idx_list = sum( [ [ii]*self.steps_per_dataset[ii] for ii in range(n) ] , [])
        while True:
            # idx = int( (i//self.batch_size) % n) # index of the current dataset in the list
            idx = idx_list[ int( i//self.batch_size  % len(idx_list) ) ]
            curr_dataset = self.dataset_list[idx]
            # print("DATA ", idx ," ", i)
            if hasattr(curr_dataset, "__getitem__"):
                '''If it's a normal Dataset, getitem and cycle increment the stored index for this specific dataset'''
                yield curr_dataset.__getitem__(intra_dataset_idxs[idx])
                intra_dataset_idxs[idx] = int( (intra_dataset_idxs[idx]+1) % dataset_lens[idx] )  
            else:
                '''If it's an Iter, we just yield the next item in it; no need to track an index'''
                yield next(curr_dataset)                  
            
            i += 1
        
    def __iter__(self):
        return iter(self.generate())















