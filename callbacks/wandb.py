from transformers.integrations import WandbCallback
import torch.nn.functional as F


class WandbAlphasCallback(WandbCallback):
    """Custom WandbCallback to log model alphas during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, freq=2):
        """Initializes the WandbAlphasCallback instance.

        Args:
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.freq = freq

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        is_gaea = model.config.gaea
        for name, w in model.named_parameters():
            if "alphas" in name:
                if is_gaea:
                    alphas = w.detach().cpu().numpy()
                else:
                    alphas = F.softmax(w.detach().cpu(), dim=0).numpy()
                labels = ["Mamba", "GPT-Neo"]  # TODO
                # data = [[label, val] for (label, val) in zip(labels, alphas)]
                for i in range(len(alphas)):
                    logs[f"{name}[{i}]"] = alphas[i]
                    # self._wandb.log({f"{name}[{i}]": alphas[i]}, "train/global_step": state.global_step)

        super().on_log(
            args=args, state=state, control=control, model=model, logs=logs, **kwargs
        )
