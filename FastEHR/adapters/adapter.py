import logging
from FastEHR.adapters.BEHRT import ConvertToBEHRT


class Adapter(object):
    """
    Wrapper for converting FastEHR datasets to various supported frameworks
    """

    @property
    def tokenizer(self):
        return self.converter.tokenizer

    def __init__(
            self,
            task:           str,
            tokenizer,
            supervised:     bool = False,
    ):
        """
        Parameters
        ----------
        task: str
            The downstream model format required. Supported formats include BEHRT.

        tokenizer:
            The FastEHR tokenizer

        supervised: bool
            Whether the FastEHR dataset is supervised or not. This is passed to
             the adapter so we know if the last token in the context needs to have
             the SEP token added.
        """

        match task.lower():
            case "behrt":
                self.converter = ConvertToBEHRT(tokenizer=tokenizer, supervised=supervised)
            case _:
                raise NotImplementedError

        logging.info(f"Adapting FastEHR{' supervised' if supervised else ''} dataloader to {task}")

    def __call__(self, data):
        return self.converter(data)
