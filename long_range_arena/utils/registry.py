optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "long_range_arena.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine":     "long_range_arena.utils.optim.schedulers.TimmCosineLRScheduler",
}

callbacks = {
    "timer":                 "long_range_arena.callbacks.timer.Timer",
    "params":                "long_range_arena.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint":      "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping":        "pytorch_lightning.callbacks.EarlyStopping",
    "swa":                   "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary":    "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar":     "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing":  "long_range_arena.callbacks.progressive_resizing.ProgressiveResizing",
    # "profiler": "pytorch_lightning.profilers.PyTorchProfiler",
}

model = {
    # Backbones from this repo
    "model":                 "long_range_arena.models.sequence.backbones.model.SequenceModel",
    "unet":                  "long_range_arena.models.sequence.backbones.unet.SequenceUNet",
    "sashimi":               "long_range_arena.models.sequence.backbones.sashimi.Sashimi",
    "sashimi_standalone":    "models.sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm":                  "long_range_arena.models.baselines.lstm.TorchLSTM",
    "gru":                   "long_range_arena.models.baselines.gru.TorchGRU",
    "unicornn":              "long_range_arena.models.baselines.unicornn.UnICORNN",
    "odelstm":               "long_range_arena.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn":          "long_range_arena.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn":            "long_range_arena.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline":   "long_range_arena.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn":             "long_range_arena.models.baselines.samplernn.SampleRNN",
    "dcgru":                 "long_range_arena.models.baselines.dcgru.DCRNNModel_classification",
    "dcgru_ss":              "long_range_arena.models.baselines.dcgru.DCRNNModel_nextTimePred",
    # Baseline CNNs
    "ckconv":                "long_range_arena.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan":               "long_range_arena.models.baselines.wavegan.WaveGANDiscriminator", # DEPRECATED
    "denseinception":        "long_range_arena.models.baselines.dense_inception.DenseInception",
    "wavenet":               "long_range_arena.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d":        "long_range_arena.models.baselines.resnet.TorchVisionResnet",  # 2D ResNet
    # Nonaka 1D CNN baselines
    "nonaka/resnet18":       "long_range_arena.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception":      "long_range_arena.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50":      "long_range_arena.models.baselines.nonaka.xresnet.xresnet1d50",
    # ViT Variants (note: small variant is taken from Tri, differs from original)
    "vit":                   "models.baselines.vit.ViT",
    "vit_s_16":              "long_range_arena.models.baselines.vit_all.vit_small_patch16_224",
    "vit_b_16":              "long_range_arena.models.baselines.vit_all.vit_base_patch16_224",
    # Timm models
    "timm/convnext_base":    "long_range_arena.models.baselines.convnext_timm.convnext_base",
    "timm/convnext_small":   "long_range_arena.models.baselines.convnext_timm.convnext_small",
    "timm/convnext_tiny":    "long_range_arena.models.baselines.convnext_timm.convnext_tiny",
    "timm/convnext_micro":   "long_range_arena.models.baselines.convnext_timm.convnext_micro",
    "timm/resnet50":         "long_range_arena.models.baselines.resnet_timm.resnet50", # Can also register many other variants in resnet_timm
    "timm/convnext_tiny_3d": "long_range_arena.models.baselines.convnext_timm.convnext3d_tiny",
    # Segmentation models
    "convnext_unet_tiny":    "long_range_arena.models.segmentation.convnext_unet.convnext_tiny_unet",
}

layer = {
    "id":         "long_range_arena.models.sequence.base.SequenceIdentity",
    "lstm":       "long_range_arena.models.baselines.lstm.TorchLSTM",
    "standalone": "models.s4.s4.S4Block",
    "s4d":        "models.s4.s4d.S4D",
    "ffn":        "long_range_arena.models.sequence.modules.ffn.FFN",
    "sru":        "long_range_arena.models.sequence.rnns.sru.SRURNN",
    "rnn":        "long_range_arena.models.sequence.rnns.rnn.RNN",  # General RNN wrapper
    "conv1d":     "long_range_arena.models.sequence.convs.conv1d.Conv1d",
    "conv2d":     "long_range_arena.models.sequence.convs.conv2d.Conv2d",
    "mha":        "long_range_arena.models.sequence.attention.mha.MultiheadAttention",
    "vit":        "long_range_arena.models.sequence.attention.mha.VitAttention",
    "performer":  "long_range_arena.models.sequence.attention.linear.Performer",
    "lssl":       "long_range_arena.models.sequence.modules.lssl.LSSL",
    "s4":         "long_range_arena.models.sequence.modules.s4block.S4Block",
    "fftconv":    "long_range_arena.models.sequence.kernels.fftconv.FFTConv",
    "s4nd":       "long_range_arena.models.sequence.modules.s4nd.S4ND",
    "mega":       "long_range_arena.models.sequence.modules.mega.MegaBlock",
    "h3":         "long_range_arena.models.sequence.experimental.h3.H3",
    "h4":         "long_range_arena.models.sequence.experimental.h4.H4",
    # 'packedrnn': 'models.sequence.rnns.packedrnn.PackedRNN',
}

layer_decay = {
    'convnext_timm_tiny': 'long_range_arena.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny',
}

model_state_hook = {
    'convnext_timm_tiny_2d_to_3d': 'long_range_arena.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d',
    'convnext_timm_tiny_s4nd_2d_to_3d': 'long_range_arena.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d',
}
