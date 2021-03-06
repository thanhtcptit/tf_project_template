import argparse


class ArgumentParserWithDefaults(argparse.ArgumentParser):
    """
    Custom argument parser that will display the default value for an argument
    in the help message.
    """

    _action_defaults_to_ignore = {
        "help", "store_true", "store_false", "store_const"}

    @staticmethod
    def _is_empty_default(default):
        if default is None:
            return True
        if isinstance(default, (str, list, tuple, set)):
            return not bool(default)
        return False

    def add_argument(self, *args, **kwargs):
        # Add default value to the help message when the default is meaningful.
        default = kwargs.get("default")
        if kwargs.get("action") not in self._action_defaults_to_ignore and \
                not self._is_empty_default(default):
            description = kwargs.get("help") or ""
            kwargs["help"] = f"{description} (default = {default})"
        super().add_argument(*args, **kwargs)


class Subcommand:
    """
    Custom object for represent sub-commands in a program.
    Read more: https://docs.python.org/3/library/argparse.html#sub-commands
    """
    def add_subparser(self, name: str, parser: argparse._SubParsersAction) \
            -> argparse.ArgumentParser:
        raise NotImplementedError()
