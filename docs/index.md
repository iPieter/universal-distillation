# Universal Language Model Distillation

```python
model = BaseTransformer("pdelobelle/robbert-v2-dutch-base", **vars(args))

logger = TensorBoardLogger("tb_logs", name="my_model")

trainer = pl.Trainer.from_argparse_args(args, logger=logger)
trainer.fit(model, train_loader)
```


::: universal_distillation.data.jit_dataloader
    rendering:
      show_root_heading: false

