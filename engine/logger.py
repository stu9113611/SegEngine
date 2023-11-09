import colored

from torch.utils.tensorboard.writer import SummaryWriter as TensorboardLogger


class Logger:
    level = 0

    @classmethod
    def info(self, category: str, msg: str) -> None:
        print(f"{colored.Fore.green}[Info] [{category}] {msg}{colored.Style.reset}")

    @classmethod
    def warn(self, category: str, msg: str) -> None:
        print(f"{colored.Fore.green}[Warn] [{category}] {msg}{colored.Style.reset}")

    @classmethod
    def error(self, category: str, msg: str) -> None:
        print(f"{colored.Fore.green}[Error] [{category}] {msg}{colored.Style.reset}")
