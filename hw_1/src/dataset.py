import os

from torchaudio.datasets import SPEECHCOMMANDS


class BinarySpeechCommands(SPEECHCOMMANDS):
    LABELS = ("yes", "no")

    def __init__(self, root: str = "./data", subset: str = "training", download: bool = True):
        super().__init__(root, download=download, subset=subset)
        self._walker = [
            w for w in self._walker
            if os.path.basename(os.path.dirname(w)) in self.LABELS
        ]
