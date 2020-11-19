from prompt_toolkit.completion import WordCompleter
import prompt_toolkit as pt

style = pt.styles.Style.from_dict({
    # User input (default text).
    '': 'white bold',
    # Prompt.
    'subcmd': '#884444',
    'path': 'ansicyan',
})

message = [
    ('class:subcmd', ''),
    ('class:sign', '> '),
]

from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit import prompt

import inspect


class CMDValidator(Validator):
    def __init__(self, api_object):
        self.api = api_object

    def validate(self, document):
        text = document.text
        splits = text.split(' ')
        if splits[0] not in self.api.functs:
            raise ValidationError(message=f'{text} function is not available',
                                  cursor_position=len(text))
        else:
            sig = inspect.signature(self.api.functs[splits[0]])
            args = text.split(',')
            if len(args) > len(sig.parameters):
                raise ValidationError(message=f'{text} function only takes {len(sig.parameters)} parameters',
                                      cursor_position=len(text))


class API:
    def __init__(self, mir_object):
        self.mir = mir_object
        cmds = [funct for funct in dir(self.mir) if not funct.startswith('_') and callable(getattr(self.mir, funct))]
        self.functs = {funct: getattr(self.mir, funct) for funct in cmds}
        self.functs['help'] = self.help
        self.functs[''] = lambda: None
        self.session = pt.shortcuts.PromptSession(
            message=message, style=style, completer=WordCompleter(list(self.functs.keys())),
            validator=CMDValidator(self)
        )

    def run(self):
        while True:
            cmd = self.session.prompt()
            splits = cmd.split(' ')
            if splits[0] in self.functs:
                args = (' '.join(str(item) for item in splits)).split(',')
                res = self.functs[splits[0]](*splits[1:])
                if res is not None:
                    print(res)

    def help(self):
        """prints help for available commands"""
        for cmd in self.functs:
            if not cmd:
                continue
            funct = self.functs[cmd]
            sig = inspect.signature(funct)
            name = f"<skyblue>{cmd}</skyblue>"
            params = []
            for par, value in sig.parameters.items():
                res = f'<green>{par}</green>'
                if value.default is not inspect.Parameter.empty:
                    res = f'{res}<darkgreen>={value.default}</darkgreen>'
                params.append(res)
            params = ', '.join(str(par) for par in params)
            doc = f'\n\t{funct.__doc__}' if funct.__doc__ else ''
            pt.print_formatted_text(pt.formatted_text.HTML(f'{name} {params}{doc}'))
