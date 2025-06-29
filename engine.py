import torch, os, json
from tqdm import tqdm
from collections import defaultdict
from regularisers import build_regulariser
from datasets import make_loaders
from torchvision.models import vgg16, squeezenet1_0
import matplotlib.pyplot as plt
import pandas as pd


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    # ---------------------------------------------------------------
    def build_model(self):
        # self.model = vgg16(num_classes=10).to(self.device)
        self.model = squeezenet1_0(num_classes=10, ).to(self.device)
        self.model.features[0] = torch.nn.Conv2d(1, 96, kernel_size=7, stride=2).to(self.device)

        # weight decay only for cfg==2
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.cfg.lr, momentum=.9,
            weight_decay=self.cfg.l2 if self.cfg.config == 2 else 0.)
        self.ce = torch.nn.CrossEntropyLoss()

        # Warm-up batches with coeff=1 to estimate scaling
        self.reg = None
        if self.cfg.config in {3, 4, 5}:
            self.reg = build_regulariser(self.cfg.config, self.model,
                                         coeff=1.0, grid=self.cfg.grid,
                                         stop_grad=self.cfg.stop_grad)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        metrics = defaultdict(float)
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            loss = self.ce(out, y)
            metrics["n"] += y.size(0)
            metrics["ce"] += loss.item() * y.size(0)
            metrics["acc"] += (out.argmax(1) == y).sum().item()
            if self.reg:
                metrics["rtd"] += self.reg().item() * y.size(0)
        for k in metrics:
            if k != "n": metrics[k] /= metrics["n"]
        return metrics

    # ---------------------------------------------------------------
    def run(self):
        train_loader, test_loader = make_loaders(
            batch_size=self.cfg.batch, few_shot=self.cfg.few,
            noise_std=self.cfg.noise)

        warmup = 2                                     # batches
        ma_ce, ma_rtd = 0., 0.

        log = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"epoch {epoch}")
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                ce_loss = self.ce(out, y)
                loss = ce_loss
                print(ce_loss.item(), end='\n')

                if self.reg:
                    rtd_loss = self.reg()
                    # right after you compute rtd_loss in the training loop:
                    if not torch.all(torch.isfinite(rtd_loss)).item():
                        print("RTD blew up!", rtd_loss)

                    # Auto-tune the coefficient AFTER warm-up
                    if epoch == 0 and i < warmup:
                        ma_ce += ce_loss.item()
                        ma_rtd += rtd_loss.item()
                        coeff = 1.
                    elif epoch == 0 and i == warmup:
                        coeff = (ma_ce / ma_rtd) * self.cfg.coef_ratio
                        print(f"[auto-scale] RTD coeff = {coeff:.4f}")
                        self.reg.coeff = coeff
                    print(rtd_loss.item(), end='\n')
                    loss = loss + rtd_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.)
                self.optimizer.step()

            # ------------- evaluation & logging --------------------
            stats = {"epoch": epoch, **self.evaluate(test_loader)}
            print(stats)
            log.append(stats)

        # -------- save CSV & plot RTD curve -------------------------
        os.makedirs(self.cfg.outdir, exist_ok=True)
        df = pd.DataFrame(log)
        csv_file = os.path.join(self.cfg.outdir, f"log_{self.cfg.tag}.csv")
        df.to_csv(csv_file, index=False)

        if "rtd" in df:
            plt.figure()
            plt.plot(df["epoch"], df["rtd"], marker='o')
            plt.title(f"RTD (test) – cfg {self.cfg.config}")
            plt.xlabel("epoch"); plt.ylabel("avg RTD per batch")
            plt.savefig(os.path.join(self.cfg.outdir, f"rtd_{self.cfg.tag}.png"))
            plt.close()
