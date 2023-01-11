import inspect

import datasketches
import numpy as np
import pandas as pd
from IPython.display import HTML, display

import lambdaprompt
import sketch


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def string_repr_truncated(val, size=30):
    result = str(val)
    if len(result) > size:
        result = result[: (size - 3)] + "..."
    return result


def get_top_n(ds, n=5, size=30, reject_all_1=True):
    top_n = [
        (count, string_repr_truncated(val, size=size))
        for val, count, *_ in ds.get_frequent_items(
            datasketches.frequent_items_error_type.NO_FALSE_POSITIVES
        )
    ][:n]
    top_n = [] if (reject_all_1 and all([c <= 1 for c, _ in top_n])) else top_n
    return {"counts": [c for c, _ in top_n], "values": [v for _, v in top_n]}


def get_distribution(ds, n=5):
    if ds.is_empty():
        return {}
    percents = np.linspace(0, 1, n)
    return {p: v for p, v in zip(percents, ds.get_quantiles(percents))}


def get_description_of_sketchpad(sketchpad):
    description = {}
    for sk in sketchpad.sketches:
        if sk.name == "Rows":
            description["rows"] = sk.data
        elif sk.name == "Count":
            description["count"] = sk.data
        elif sk.name == "DS_THETA":
            description["uniqecount-est"] = sk.data.get_estimate()
        elif sk.name == "UnicodeMatches":
            description["unicode"] = sk.data
        elif sk.name == "DS_FI":
            description["top-n"] = get_top_n(sk.data)
        elif sk.name == "DS_KLL":
            description["quantiles"] = get_distribution(sk.data)
    return description


def get_description_from_parts(
    column_names, data_types, extra_information, index_col_name=None
):
    descriptions = []
    for colname, dtype, extra in zip(column_names, data_types, extra_information):
        description = {
            "column-name": colname,
            "type": dtype,
            "index": colname == index_col_name,
        }
        if not isinstance(extra, sketch.SketchPad):
            # try and load it as a sketchpad
            try:
                if "version" in extra:
                    extra = sketch.SketchPad.from_dict(extra)
            except:
                pass
        if isinstance(extra, sketch.SketchPad):
            extra = get_description_of_sketchpad(extra)
        description.update(extra)
        descriptions.append(description)
    return descriptions


def get_parts_from_df(df, useSketches=False):
    index_col_name = df.index.name
    df = df.reset_index()
    column_names = df.columns
    data_types = df.dtypes
    if useSketches:
        extras = list(sketch.Portfolio.from_dataframe(df).sketchpads.values())
        # extras = [get_description_of_sketchpad(sketchpad) for sketchpad in sketchpads]
    else:
        extras = []
        for col in df.columns:
            extra = {
                "rows": len(df[col]),
                "head-sample": df[col].head(5).tolist(),
                "count": df[col].count(),
                "uniqecount": df[col].nunique(),
            }
            # if column is numeric, get quantiles
            if df[col].dtype in [np.float64, np.int64]:
                extra["quantiles"] = df[col].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            extras.append(extra)
    return column_names, data_types, extras, index_col_name


def call_prompt_on_dataframe(df, prompt, **kwargs):
    names = retrieve_name(df)
    name = "df" if len(names) == 0 else names[0]
    column_names, data_types, extras, index_col_name = get_parts_from_df(df)
    # this is what will be proxied either to sketch or directly through lambdaprompt
    text_to_copy = prompt(
        dfname=name,
        column_names=column_names,
        data_types=data_types,
        extras=extras,
        index_col_name=index_col_name,
        **kwargs,
    )
    return text_to_copy


howto_prompt = lambdaprompt.GPT3Prompt(
    """
For the pandas dataframe ({{ dfname }}) the user wants code to solve a problem.
Summary statistics and descriptive data of dataframe [`{{ dfname }}`]:
```
{{ data_description }}
```

Code to solve [ {{ how }} ]?:
```python
""",
    stop=["```"],
    # model_name="code-davinci-002",
)


@lambdaprompt.prompt
def howto_from_parts(dfname, column_names, data_types, extras, index_col_name, how):
    description = get_description_from_parts(
        column_names, data_types, extras, index_col_name
    )
    description = pd.json_normalize(description).to_csv(index=False)
    return howto_prompt(dfname=dfname, data_description=description, how=how)


ask_prompt = lambdaprompt.GPT3Prompt(
    """
For the pandas dataframe ({{ dfname }}) the user wants an answer to a question about the data.
Summary statistics and descriptive data of dataframe [`{{ dfname }}`]:
```
{{ data_description }}
```

{{ question }}
Answer:
```
""",
    stop=["```"],
)


@lambdaprompt.prompt
def ask_from_parts(dfname, column_names, data_types, extras, index_col_name, question):
    description = get_description_from_parts(
        column_names, data_types, extras, index_col_name
    )
    description = pd.json_normalize(description).to_csv(index=False)
    return ask_prompt(dfname=dfname, data_description=description, question=question)


@pd.api.extensions.register_dataframe_accessor("sketch")
class SketchHelper:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def howto(self, how, call_display=True):
        result = call_prompt_on_dataframe(self._obj, howto_from_parts, how=how)
        if not call_display:
            return result
        display(HTML(f"""<pre>{result}</pre>"""))

    def ask(self, question, call_display=True):
        result = call_prompt_on_dataframe(self._obj, ask_from_parts, question=question)
        if not call_display:
            return result
        display(HTML(f"""<pre>{result}</pre>"""))

    def apply(self, prompt_template_string):
        new_gpt3_prompt = lambdaprompt.GPT3Prompt(prompt_template_string)
        pass
