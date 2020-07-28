import argparse


parser = argparse.ArgumentParser(description="SGD/SWA training")


parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)


args = parser.parse_args()


print(args.model)

