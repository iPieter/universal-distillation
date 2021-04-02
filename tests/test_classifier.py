from pytorch_lightning import Trainer, seed_everything
from universal_distillation.data.jit_data_module import JITDataModule
from universal_distillation.modules.base import BaseTransformer
from transformers import AutoTokenizer
import tempfile
import os
import pytorch_lightning as pl

SAMPLE = """Mr President, concerning the Minutes, item 2: yesterday I brought up the subject of Norwegian salmon dumping.
I have to report to the House that the reports we heard yesterday seem to be true that Norway has done a deal with the European Union which will involve a voluntary code of export limitation.
There will be much anger within the EU that the country which voted No to membership of the club seems to have better access to the Commission than we in this Parliament when we tried to raise the matter yesterday.
Mr President, ladies and gentlemen, please join the members of the Austrian Freedom Party in welcoming to the gallery a very special guest.
Mr Lama Gangchen, the President of the World Peace Foundation, who is visiting the European Parliament for the first time.
Thank you!"""


def test_classifier():

    with tempfile.TemporaryDirectory() as tempdir:
        tmpfilepath = os.path.join(tempdir, "data.txt")
        with open(tmpfilepath, "w") as tmpfile:
            tmpfile.write(SAMPLE)

        # This is a reaaally small model and perfect for this test
        model_string = "sshleifer/tiny-distilroberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_string)

        data_module = JITDataModule(
            file_path=tmpfilepath,
            tokenizer=tokenizer,
        )

        # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
        # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
        # dataset_train, dataset_val = random_split(dataset, [int(len(dataset)*0.9), int(len(dataset)*0.1)])

        model = BaseTransformer(model_string, train_batch_size=2)

        trainer = pl.Trainer()
        trainer.fit(model, data_module)

        assert True
