import argparse
import localvglobal.models as models
from localvglobal.data import loaders

parser = argparse.ArgumentParser(description="SGD/SWA training")

parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    metavar="DIRECTORY",
    help="path to experiment directory (default: None)",
)

parser.add_argument(
    "--name",
    type=str,
    default=None,
    required=True,
    metavar="NAME",
    help="name to give to saved model (default: None)",
)

parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="name of model class (default: None)",
)

parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    required=True,
    metavar="DATASET",
    help="name of dataset (default: None)",
)

parser.add_argument(
    "--no_validation",
    action="store_true",
    help="don't validate (i.e. use all data for training)",
)

parser.add_argument(
    "--val_ratio",
    type=float,
    default=0.2,
    required=False,
    metavar="RATIO",
    help="ratio of dataset to use for validation (default: 0.2, ignored if no_validation is True)",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    required=False,
    metavar="BATCH_SIZE",
    help="number of examples in a batch (default: 128)",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="show live training progress",
)

parser.add_argument(
    "--save_graph",
    action="store_true",
    help="save final training graph",
)

args = parser.parse_args()
print(':::')
print('HELLO!')
print(':::')
print(args)
#break
# parse model class
args.model = getattr(models, args.modell)

# parse data loaders
loader = loaders[args.dataset](dir=args.dir, )
args.train_loader = loader['train']
args.train_loader = loader['valid']


use_validation=True,
val_size=10000,
train_transforms=DEFAULT_TRANSFORM,
test_transforms=DEFAULT_TRANSFORM,
pin_memory=True,
batch_size=128,
num_workers=1


