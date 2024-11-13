import torch
from .tasks import get_task_sampler
from .samplers import get_data_sampler
from .curriculum import Curriculum

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def _combine(xs_b, ys_b):
    """Interleaves the x's and the y's into a single sequence.
    (B, L, D)
    Even indices L: xs; 
    Odd indices L: ys; since ground truth are scalar only, only index D0 is populated, rest are 0 values 
    """
    bsize, points, dim = xs_b.shape
    ys_b_wide = torch.cat(
        (
            ys_b.view(bsize, points, 1),
            torch.zeros(bsize, points, dim - 1, device=ys_b.device),
        ),
        axis=2,
    )
    zs = torch.stack((xs_b, ys_b_wide), dim=2)
    zs = zs.view(bsize, 2 * points, dim) # (B, L, D)
    return zs

class RegressionIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, task_name, n_dims, n_dims_truncated,
                       batch_size, n_points,
                       x_bias=None, x_scale=None,
                       task_kwargs:dict={}, task_sampler_kwargs:dict={}, curriculum=None):
        '''NOTE: batch_size in this dataset should equal the batch_size used in the dataloader/HF trainer'''
        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated
        self.batch_size = batch_size
        self.n_points = n_points
        self.x_bias = x_bias
        self.x_scale = x_scale
        self.task_name = task_name
        self.task_kwargs = task_kwargs
        self.task_sampler_kwargs = task_sampler_kwargs
        self.data_sampler = get_data_sampler("gaussian", self.n_dims, bias=self.x_bias, scale=self.x_scale)
        self.task_sampler = get_task_sampler(task_name=self.task_name, n_dims=self.n_dims, batch_size=1, 
                                             pool_dict=None, num_tasks=None, **task_kwargs)
        self.curriculum = curriculum
    def generate(self):
        i=0
        has_curriculum = self.curriculum is not None
        while True:
            if i==0:
                # finished a batch step; reset the task w 
                # print("reset task w")
                if "sparse" in self.task_name:
                    self.task_sampler_kwargs["valid_coords"] = self.curriculum.n_dims_truncated if has_curriculum else self.n_dims_truncated
                task = self.task_sampler(**self.task_sampler_kwargs)
                # Generate a batch of data
                xs_batch = self.data_sampler.sample_xs(
                    n_points= self.curriculum.n_points if has_curriculum else self.n_points, 
                    b_size=self.batch_size, # one full batch, return them one at a time
                    n_dims_truncated= self.curriculum.n_dims_truncated if has_curriculum else self.n_dims_truncated, 
                    seeds=None
                )
            xs = xs_batch[i:i+1]
    
            if has_curriculum and i+1==self.batch_size:
                self.curriculum.update()
                # print(self.curriculum.n_dims_truncated, self.curriculum.n_points)
            ys = task.evaluate(xs)
            zs = _combine(xs, ys)
            zs = zs[:,:-1,:] # (B, 2*n_pts, n_dim) -> (B, 2*n_pts-1, n_dim)
            ys = ys[:,-1] #  -> (B,) just last point
            
            i = (i+1) % self.batch_size

            yield {"input_ids":zs.squeeze(0), 
                   "labels":ys.squeeze(0)}
        
    def __iter__(self):
        return iter(self.generate())
class RegressionCurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, n_dims, n_dims_truncated,
                       batch_size, n_points,
                       x_bias=None, x_scale=None,
                       task_kwargs:dict={}, task_sampler_kwargs:dict={}, curriculum=None,
                       n_samples = 1_000_000,
                ):
        '''
        Generates data the same way as RegressionIterDataset, but without a generator (generator breaks ASHA HPO)
        n_samples is purely a placeholder returned by __len__ and does not actually affect the number of unique datapoints in the dataset
            Regression will be trained by steps instead of epochs anyway
        NOTE: batch_size in this dataset should equal the batch_size used in the dataloader/HF trainer
        '''
        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated
        self.batch_size = batch_size
        self.n_points = n_points
        self.x_bias = x_bias
        self.x_scale = x_scale
        self.task_name = task_name
        self.task_kwargs = task_kwargs
        self.task_sampler_kwargs = task_sampler_kwargs
        self.data_sampler = get_data_sampler("gaussian", self.n_dims, bias=self.x_bias, scale=self.x_scale)
        self.task_sampler = get_task_sampler(task_name=self.task_name, n_dims=self.n_dims, batch_size=1, 
                                             pool_dict=None, num_tasks=None, **task_kwargs)
        self.curriculum = curriculum
        self.n_samples = n_samples
        self.i = 0 # analogous to the i from RegressionIterDataset generate()

        self.task = None 
        self.xs_batch = None
    
    def __len__(self):
        return self.n_samples
    def __getitem__(self, trash_i):
        # The argument trash_i does not matter; curriculum is updated and data is returned based only on class instance internal self.i
        # During training for a non-iterable dataset the index passed to getitem will be in random order anyway
        has_curriculum = self.curriculum is not None
        if self.i==0:
            # finished a batch step; reset the task w 
            if "sparse" in self.task_name:
                self.task_sampler_kwargs["valid_coords"] = self.curriculum.n_dims_truncated if has_curriculum else self.n_dims_truncated
            self.task = self.task_sampler(**self.task_sampler_kwargs)
            # Generate a batch of data
            self.xs_batch = self.data_sampler.sample_xs(
                n_points= self.curriculum.n_points if has_curriculum else self.n_points, 
                b_size=self.batch_size, # one full batch, return them one at a time
                n_dims_truncated= self.curriculum.n_dims_truncated if has_curriculum else self.n_dims_truncated, 
                seeds=None
            )
        xs = self.xs_batch[self.i:self.i+1]

        if has_curriculum and self.i+1==self.batch_size:
            self.curriculum.update()
        ys = self.task.evaluate(xs)
        zs = _combine(xs, ys)
        zs = zs[:,:-1,:] # (B, 2*n_pts, n_dim) -> (B, 2*n_pts-1, n_dim)
        ys = ys[:,-1] #  -> (B,) just last point
        
        self.i = (self.i+1) % self.batch_size

        return {"input_ids":zs.squeeze(0), 
               "labels":ys.squeeze(0)}

# this is a RegressionIterDataset with no curriculum 
class RegressionDataset(torch.utils.data.Dataset): 
    def __init__(self, task_name, n_dims, n_dims_truncated,
                       batch_size, n_points,
                       x_bias=None, x_scale=None,
                       task_kwargs:dict={}, task_sampler_kwargs:dict={}, curriculum=None,
                       n_samples=1_000_000):
        self.n_samples = n_samples
        self.iter_dataset = RegressionIterDataset(
            task_name, n_dims, n_dims_truncated,
            batch_size, n_points,
            x_bias, x_scale,
            task_kwargs, task_sampler_kwargs, curriculum=None
        )

        self.n_dims = n_dims
        self.n_dims_truncated = n_dims_truncated
        self.batch_size = batch_size
        self.n_points = n_points
        self.x_bias = x_bias
        self.x_scale = x_scale
        self.task_name = task_name
        self.task_kwargs = task_kwargs
        self.task_sampler_kwargs = task_sampler_kwargs
        self.data_sampler = self.iter_dataset.data_sampler
        self.task_sampler = self.iter_dataset.task_sampler
        self.curriculum = curriculum

        self.data_it = self.iter_dataset.__iter__()
    def __len__(self):
        return self.n_samples
    def __getitem__(self,i):
        return next(self.data_it)
        

def get_regression_dataset(task_name, n_dims=20, n_dims_truncated=5,
                           batch_size=64, n_points=41,
                           n_samples=None):
    if task_name=="linear_regression":
        task_kwargs = {}
        task_sampler_kwargs = {}
        curriculum = Curriculum(5,20,1,2000,
                                11,41,2,2000)
    elif task_name=="noisy_linear_regression":
        task_kwargs = {"noise_std": 1}
        task_sampler_kwargs = {}
        curriculum = Curriculum(5,20,1,2000,
                                11,41,2,2000)
    elif task_name=="quadratic_regression": # the curriculum might by ill-suited
        task_kwargs = {}
        task_sampler_kwargs = {}
        curriculum = Curriculum(5,20,1,2000,
                                11,41,2,2000)
    elif task_name=="sparse_linear_regression":
        task_kwargs = {"sparsity": 3}
        task_sampler_kwargs = {"valid_coords": n_dims_truncated}
        curriculum = Curriculum(5,20,1,2000,
                                11,41,2,2000)
    elif task_name=="relu_2nn_regression":
        task_kwargs = {"hidden_layer_size": 100}
        task_sampler_kwargs = {}
        curriculum = Curriculum(5,20,1,2000,
                                26,101,5,2000)
    elif task_name=="decision_tree":
        task_kwargs = {"depth": 4}
        task_sampler_kwargs = {}
        curriculum = Curriculum(5,20,1,2000,
                                26,101,5,2000)
    else: 
        raise NotImplementedError

    if n_samples is None:
        # return RegressionIterDataset(task_name, n_dims, n_dims_truncated,
        #                              batch_size, n_points, 
        #                              x_bias=None, x_scale=None,
        #                              task_kwargs=task_kwargs, task_sampler_kwargs=task_sampler_kwargs, curriculum=curriculum)
        return RegressionCurriculumDataset(
            task_name, n_dims, n_dims_truncated,
            batch_size, n_points, 
            x_bias=None, x_scale=None,
            task_kwargs=task_kwargs, task_sampler_kwargs=task_sampler_kwargs, curriculum=curriculum
        )
    else:
        # return RegressionDataset(task_name, curriculum.dims_end, curriculum.dims_end,
        #                          batch_size, curriculum.points_end, 
        #                          x_bias=None, x_scale=None,
        #                          task_kwargs=task_kwargs, task_sampler_kwargs=task_sampler_kwargs,curriculum=None,
        #                          n_samples=n_samples)
        return RegressionCurriculumDataset(
            task_name, curriculum.dims_end, curriculum.dims_end,
            batch_size, curriculum.points_end, 
            x_bias=None, x_scale=None,
            task_kwargs=task_kwargs, task_sampler_kwargs=task_sampler_kwargs,curriculum=None,
            n_samples=n_samples
        )
    '''
    GENERATOR OBJECTS IN THE DATASET/HF TRAINER BREAK RAYTUNE
    Replace all returned datasets with the RegressionCurriculumDataset which extends regular torch dataset and does not have generators
    '''
    