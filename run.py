import argparse, itertools, yaml, random, numpy as np, torch, os
from engine import Engine


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=int, default=1,
                   choices=[1, 2, 3, 4, 5])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--few", type=int, default=None,
                   help="few-shot samples per class (None = full data)")
    p.add_argument("--noise", type=float, default=0.0,
                   help="σ of Gaussian noise added to every pixel")
    p.add_argument("--grid", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--stop_grad", choices=[None, "before", "after"],
                   default=None)
    return p.parse_args()


def fix_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


if __name__ == "__main__":
    args = parse()
    results_dir = f"results/cfg{args.config}"
    os.makedirs(results_dir, exist_ok=True)

    for run in range(args.seeds):
        fix_seed(run)
        cfg = argparse.Namespace(
            config=args.config,
            lr=0.01,
            l2=5e-4,
            coef_ratio=0.01,            # “slightly less than CE”
            epochs=5,
            batch=args.batch,
            few=args.few,
            noise=args.noise,
            grid=args.grid,
            stop_grad=args.stop_grad,
            outdir=results_dir,
            tag=f"s{run}"
        )
        Engine(cfg).run()
