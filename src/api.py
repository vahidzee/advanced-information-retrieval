from prompt_toolkit.completion import Completion, Completer
import prompt_toolkit as pt
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from .utils import auto_cast
import inspect

style = pt.styles.Style.from_dict({
    # User input (default text).
    '': 'white bold',
    # Prompt.
    'subcmd': '#884444',
    'sign': 'ansicyan',
})

message = [
    ('class:subcmd', ''),
    ('class:sign', '> '),
]


def repr_default_value(value):
    if isinstance(value, str):
        return f'"{value}"'
    return value


class CMDValidator(Validator):
    def __init__(self, api_object):
        self.api = api_object

    def validate(self, document):
        text = document.text
        splits = text.split(' ')
        if splits[0] not in self.api.functs:
            raise ValidationError(message=f'command {text} is not available',
                                  cursor_position=len(text))
        else:
            sig = inspect.signature(self.api.functs[splits[0]])
            args = text.split(',')
            if len(args) - 1 > len(sig.parameters):
                raise ValidationError(message=f'command {text} only takes {len(sig.parameters)} parameters',
                                      cursor_position=len(text))


class SuggestParameter(AutoSuggest):
    def __init__(self, api_object):
        self.api = api_object

    def get_suggestion(self, buffer, document):
        text = document.text
        splits = text.split(' ')
        args = (' '.join(splits[1:])).split(',')
        if splits[0] not in self.api.functs or len(splits) == 1:
            return None
        else:
            sig = inspect.signature(self.api.functs[splits[0]])
            last_arg = args[-1].strip()
            start = len(args) - 1 if not last_arg else len(args)
            params = [
                f'{par}={repr_default_value(value.default)}' if value.default is not inspect.Parameter.empty \
                    else f'[{par}]' for par, value in sig.parameters.items()
            ]
            params = f"{', ' if last_arg else ''}{', '.join(param for param in params[start:])}"
            return Suggestion(params)


class APICompleter(Completer):
    def __init__(
            self,
            api_object,
            ignore_case: bool = False,
            meta_dict=None,
            WORD: bool = False,
            sentence: bool = False,
            match_middle: bool = False,
            pattern=None,
    ) -> None:

        assert not (WORD and sentence)
        self.api = api_object
        self.ignore_case = ignore_case
        self.meta_dict = meta_dict or {}
        self.WORD = WORD
        self.sentence = sentence
        self.match_middle = match_middle
        self.pattern = pattern

    def _get_words(self, document):
        splits = document.split(' ')
        if len(splits) == 1:
            return list(self.api.functs.keys())
        cmd = splits[0]
        args = (' '.join(str(item) for item in splits[1:])).split(', ')
        if not args[-1]:
            del args[-1]
        if cmd in self.api.suggestors:
            return self.api.suggestors[cmd](args)
        return []

    def get_completions(self, document, complete_event):
        # Get list of words.
        words = self._get_words(document.text)

        # Get word/text before cursor.
        if self.sentence:
            word_before_cursor = document.text_before_cursor
        else:
            word_before_cursor = document.get_word_before_cursor(
                WORD=self.WORD, pattern=self.pattern
            )

        if self.ignore_case:
            word_before_cursor = word_before_cursor.lower()

        def word_matches(word: str) -> bool:
            if self.ignore_case:
                word = word.lower()

            if self.match_middle:
                return word_before_cursor in word
            else:
                return word.startswith(word_before_cursor)

        for a in words:
            if word_matches(a):
                display_meta = self.meta_dict.get(a, "")
                yield Completion(a, -len(word_before_cursor), display_meta=display_meta)


class API:
    def __init__(self, mir_object):
        self.mir = mir_object
        cmds = [funct for funct in dir(self.mir) if not funct.startswith('_') and callable(getattr(self.mir, funct))]
        self.functs = {funct: getattr(self.mir, funct) for funct in cmds if not funct.endswith('suggestion')}
        self.suggestors = {funct.replace('_suggestion', ''): getattr(self.mir, funct) for funct in cmds if
                           funct.endswith('suggestion')}
        self.functs['help'] = self.help
        self.functs[''] = lambda: None
        self.functs['exit'] = exec
        self.suggestors['help'] = self.help_suggestion
        self.session = pt.shortcuts.PromptSession(
            message=message, style=style, completer=APICompleter(self),
            validator=CMDValidator(self), auto_suggest=SuggestParameter(self)
        )

    def run(self):
        while True:
            cmd = self.session.prompt()
            splits = cmd.split(' ')
            if splits[0] in self.functs:
                args = (' '.join(str(item) for item in splits[1:])).split(', ')
                if not args[-1]:
                    del args[-1]
                args = [auto_cast(arg) for arg in args]
                res = self.functs[splits[0]](*args)
                if res is not None:
                    print(res)

    def help_suggestion(self, args):
        if not args or not args[0]:
            return self.functs.keys()
        if len(args) > 1:
            return []
        return filter(lambda x: x.startswith(args[0]), self.functs.keys())

    def help(self, cmd: str = None):
        """prints help for available commands or the specified command"""
        for cmd in self.functs if cmd is None else [cmd]:
            if not cmd:
                continue
            funct = self.functs[cmd]
            sig = inspect.signature(funct)
            name = f"<skyblue>{cmd}</skyblue>"
            params = []
            for par, value in sig.parameters.items():
                res = f'<green>{par}</green>'
                if value.annotation is not inspect.Parameter.empty:
                    res = f'{res}<gray>:{value.annotation.__name__}</gray>'
                if value.default is not inspect.Parameter.empty:
                    res = f'{res}<darkgray>={repr_default_value(value.default)}</darkgray>'
                params.append(res)
            params = ', '.join(str(par) for par in params)
            doc = f'\n\t{funct.__doc__}' if funct.__doc__ else ''
            pt.print_formatted_text(pt.formatted_text.HTML(f'{name}'), end='')
            pt.print_formatted_text(pt.formatted_text.HTML(f' {params}'), end='')
            pt.print_formatted_text(pt.formatted_text.HTML(f'{doc}'))
