import data

from argparse import ArgumentParser
from dataclasses import dataclass, field

from hyperparams import Hyperparameter, Numerical, Categorial
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from functools import partial


METHOD_TO_HYPERPARAMS = {
    SVC: {
        'gamma': Numerical('float', 1e-3, 1e1, is_log_scale=True),
        'C': Numerical('int', 1e5, 1e9, is_log_scale=True),
        'kernel': Categorial('poly', 'rbf', 'sigmoid')
    },
    
    XGBRegressor: {
        'gamma': Numerical('float', 0.2, 0.3),
        'learning_rate': Numerical('float', 0.2, 0.4),
        'booster': Categorial('gblinear', 'gbtree', 'dart')
    }
}


NAME_TO_DATASET = {
    'balance': data.Balance,
    'bank-marketing': data.BankMarketing,
    'banknote': data.Banknote,
    'breast-cancer': data.BreastCancer,
    'car-evaluation': data.CarEvaluation,
    'cnae9': data.CNAE9,
    'credit-approval': data.CreditApproval,
    'digits': data.Digits,
    'ecoli': data.Ecoli,
    'parkinsons': data.Parkinsons,
    'semeion': data.Semeion,
    'statlog-segmentation': data.StatlogSegmentation,
    'wilt': data.Wilt,
    'zoo': data.Zoo,
    'transformator': data.Transformator,
    'turbine': data.Turbine
}

NAME_TO_ESTIMATOR = {
    'svc': partial(SVC, max_iter=1000),
    'svr': partial(SVR, max_iter=1000),
    
    'xgbclassifier': partial(XGBClassifier, n_jobs=1),
    'xgbregressor': partial(XGBRegressor, n_jobs=1),
    
    
    'mlpclassifier': MLPClassifier,
    'mlpregressor': MLPRegressor
}


@dataclass
class ConsoleArgument:
    max_iter: int
    estimator: SVC | XGBClassifier | MLPClassifier | XGBRegressor
    dataset: data.Dataset
    hyperparams: Hyperparameter = field(init=False)
    dir: str
    trials: int
    n_jobs: int
    
    iopt_npp: int

    def __post_init__(self):
        estimator = self.estimator
        if isinstance(estimator, partial):
            estimator = estimator.func
        self.hyperparams = METHOD_TO_HYPERPARAMS[estimator]


def get_estimator(name: str) -> SVC | XGBClassifier | MLPClassifier:
    try:
        return NAME_TO_ESTIMATOR[name]
    except:
        raise ValueError(f'Estimator "{name}" do not support')


def get_datasets(names: str) -> data.Dataset:
    try:
        result = []
        for x in names:
            result.append(NAME_TO_DATASET[x])
        return result
    except KeyError:
        raise ValueError(f' Dataset "{x}" do not support')


def parse_arguments():
    """
    --max-iter:
        int, positive
    --dataset:
        names of dataset, see all names in NAME_TO_DATASET dict
    --method:
        must be or svc, or xgb, or mlp
    --dir:
        name of the dir to save the results (result by default)

    """
    parser = ArgumentParser()
    parser.add_argument('--max-iter', type=int)
    parser.add_argument('--dataset', nargs='*')
    parser.add_argument('--method')
    parser.add_argument('--dir', default='result')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--iopt_npp', type=int, default=1)
    
    args = parser.parse_args()
    assert args.max_iter > 0, 'Max iter must be positive'
    assert args.trials > 0, 'Trials must be positive'
    assert args.n_jobs > 0, 'n_jobs must be positive'
    assert args.iopt_npp > 0

    return ConsoleArgument(args.max_iter,
                           get_estimator(args.method),
                           get_datasets(args.dataset),
                           args.dir, args.trials, args.n_jobs, args.iopt_npp)
