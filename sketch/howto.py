import inspect

import numpy as np
import pandas as pd
from IPython.display import HTML, display

import lambdaprompt
import sketch


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


howto_prompt = lambdaprompt.GPT3Prompt(
    """
For the pandas dataframe ({{ dfname }}) the user wants code to solve a problem.
First 2 rows of `{{ dfname }}`:
```
{{ firsttworows }}
```
Summary statistics of `{{ dfname }}`:
```
{{ summary }}
```
Code to solve [ {{ how }} ]?
```python
""",
    stop=["```"],
    model_name="code-davinci-002",
)


@pd.api.extensions.register_dataframe_accessor("sketch")
class SketchHelper:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def howto(self, how):
        names = retrieve_name(self._obj)
        name = "df" if len(names) == 0 else names[0]
        summary = self._obj.describe().__repr__()
        firsttworows = self._obj.head(2).__repr__()
        text_to_copy = howto_prompt(
            dfname=name, firsttworows=firsttworows, summary=summary, how=how
        )
        # <button onclick='navigator.clipboard.writeText("{text_to_copy}");'>Copy</button>
        to_display = f"""<pre>{text_to_copy}</pre>"""
        display(HTML(to_display))
