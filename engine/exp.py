import os

from engine.logger import Logger
from shutil import copy


class ExpDir:
    def __init__(self, name: str, logdir: str = "./log") -> None:
        self.version = self.get_version(logdir, name)
        Logger.info("ExpDir", f"version: {self.version}")
        self.root = os.path.join(logdir, name, f"v_{self.version}")
        self.checkpoint = os.path.join(self.root, "checkpoints")
        self.train_images = os.path.join(self.root, "train_images")
        self.val_images = os.path.join(self.root, "val_images")
        if not self.version:
            os.makedirs(os.path.join(logdir, name))
        os.makedirs(self.root)
        os.makedirs(self.checkpoint)
        os.makedirs(self.train_images)
        os.makedirs(self.val_images)

    def backup_script(self, filename: str):
        print("file backuped:", os.path.basename(filename))
        copy(filename, os.path.join(self.root, "backup_" + os.path.basename(filename)))

    def get_version(self, logdir: str, name: str) -> int:
        version = 0
        if os.path.exists(os.path.join(logdir, name)):
            for s in os.listdir(os.path.join(logdir, name)):
                if version <= int(s.split("_")[-1]):
                    version = int(s.split("_")[-1]) + 1
        return version
