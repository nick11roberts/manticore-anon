import math

class Curriculum:
    # def __init__(self, args):
    def __init__(self, dims_start,  dims_end,  dims_inc, dims_interval,
                       points_start,  points_end, points_inc, points_interval):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        # self.n_dims_truncated = args.dims.start
        # self.n_points = args.points.start
        # self.n_dims_schedule = args.dims
        # self.n_points_schedule = args.points
        # self.step_count = 0

        self.n_dims_truncated = dims_start
        self.dims_interval = dims_interval
        self.dims_inc = dims_inc
        self.dims_end = dims_end
        
        self.n_points = points_start
        self.points_interval = points_interval
        self.points_inc = points_inc
        self.points_end = points_end
        
        self.step_count = 0

    def update(self):
        self.step_count += 1
        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.dims_interval, self.dims_inc, self.dims_end
        )
        self.n_points = self.update_var(self.n_points, self.points_interval, self.points_inc, self.points_end)

    def update_var(self, var, interval, inc, end):
        if self.step_count % interval == 0:
            var += inc

        return min(var, end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)