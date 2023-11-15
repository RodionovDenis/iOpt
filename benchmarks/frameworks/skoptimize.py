import skopt

from hyperparams import Numerical, Categorial
from .interface import Searcher, Point

class SkoptSearcher(Searcher):
    def __init__(self, max_iter, *, is_deterministic=False):
        super().__init__(framework_name='Scikit-Optimize',
                         max_iter=max_iter,
                         is_deterministic=is_deterministic)
    
    def _get_points(self):
        space = self.__get_hyperparam_space()
        points = []
        
        @skopt.utils.use_named_args(space)
        def objective(**params):
            point = self._calculate_metric(params)
            points.append(point)
            return -point.value

        skopt.gp_minimize(objective, space, n_calls=self.max_iter)
        return points

    def _get_searcher_params(self):
        return {}
    
    def __get_hyperparam_space(self):
        space = []
        func = {'int': skopt.space.Integer,
                'float': skopt.space.Real}
        
        for name, p in self.hyperparams.items():
            if isinstance(p, Numerical):
                param = func[p.type](p.min_value, p.max_value, 
                                     prior='log-uniform' if p.is_log_scale else 'uniform',
                                     name=name)
            elif isinstance(p, Categorial):
                param = skopt.space.Categorical(p.values, name=name)
            space.append(param)
        return space
