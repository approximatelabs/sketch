import ast
import base64
import importlib
import inspect
import json
import logging
import os
import uuid

import datasketches
import numpy as np
import pandas as pd
import requests
from IPython.display import HTML, display

import lambdaprompt
import sketch


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).
    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return 1
    elif val in ("n", "no", "f", "false", "off", "0"):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))


def string_repr_truncated(val, size=100):
    result = str(val)
    if len(result) > size:
        result = result[: (size - 3)] + "..."
    return result


def get_top_n(ds, n=5, size=100, reject_all_1=True):
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
    column_names = [str(x) for x in df.columns]
    data_types = [str(x) for x in df.dtypes]
    if useSketches:
        extras = list(sketch.Portfolio.from_dataframe(df).sketchpads.values())
        # extras = [get_description_of_sketchpad(sketchpad) for sketchpad in sketchpads]
    else:
        extras = []
        for col in df.columns:
            extra = {
                "rows": len(df[col]),
                "count": int(df[col].count()),
                "uniquecount": int(df[col].apply(str).nunique()),
                "head-sample": str(
                    [string_repr_truncated(x) for x in df[col].head(5).tolist()]
                ),
            }
            # if column is numeric, get quantiles
            if df[col].dtype in [np.float64, np.int64]:
                extra["quantiles"] = str(
                    df[col].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
                )
            extras.append(extra)
    return column_names, data_types, extras, index_col_name


def to_b64(data):
    return base64.b64encode(json.dumps(data).encode("utf-8")).decode("utf-8")


def from_b64(data):
    return json.loads(base64.b64decode(data.encode("utf-8")).decode("utf-8"))


def call_prompt_on_dataframe(df, prompt, **kwargs):
    names = retrieve_name(df)
    name = "df" if len(names) == 0 else names[0]
    column_names, data_types, extras, index_col_name = get_parts_from_df(df)
    max_columns = int(os.environ.get("SKETCH_MAX_COLUMNS", "20"))
    if len(column_names) > max_columns:
        raise ValueError(
            f"Too many columns ({len(column_names)}), max is {max_columns} in current version (set SKETCH_MAX_COLUMNS to override)"
        )
    prompt_kwargs = dict(
        dfname=name,
        column_names=to_b64(column_names),
        data_types=to_b64(data_types),
        extras=to_b64(extras),
        index_col_name=index_col_name,
        **kwargs,
    )
    # We now have all of our vars, let's decide if we use an external service or local prompt
    if strtobool(os.environ.get("SKETCH_USE_REMOTE_LAMBDAPROMPT", "True")):
        url = os.environ.get("SKETCH_ENDPOINT_URL", "https://prompts.approx.dev")
        try:
            response = requests.get(
                f"{url}/prompt/{prompt.name}",
                params=prompt_kwargs,
            )
            response.raise_for_status()
            text_to_copy = response.json()
        except Exception as e:
            print(
                f"""Failed to use remote {url}.. {str(e)}. 
Consider setting SKETCH_USE_REMOTE_LAMBDAPROMPT=False 
and run with your own open-ai key
"""
            )
            text_to_copy = f"SKETCH ERROR - see print logs for full error"
    else:
        # using local version
        text_to_copy = prompt(**prompt_kwargs)
    return text_to_copy


howto_prompt = lambdaprompt.Completion(
    """
For the pandas dataframe ({{ dfname }}) the user wants code to solve a problem.
Summary statistics and descriptive data of dataframe [`{{ dfname }}`]:
```
{{ data_description }}
```
The dataframe is loaded and in memory, and currently named [ {{ dfname }} ].

Code to solve [ {{ how }} ]?:
```python
{% if previous_answer is defined %}
{{ previous_answer }}
```
{{ previous_error }}

Fixing for error, and trying again...
Code to solve [ {{ how }} ]?:
```
{% endif %}
""",
    stop=["```"],
    # model_name="code-davinci-002",
)


@lambdaprompt.prompt
def howto_from_parts(
    dfname, column_names, data_types, extras, how, index_col_name=None
):
    column_names = from_b64(column_names)
    data_types = from_b64(data_types)
    extras = from_b64(extras)
    description = get_description_from_parts(
        column_names, data_types, extras, index_col_name
    )
    description = pd.json_normalize(description).to_csv(index=False)
    code = howto_prompt(dfname=dfname, data_description=description, how=how)
    try:
        ast.parse(code)
    except SyntaxError as e:
        # if we get a syntax error, try again, but include the error message
        # only do 1 retry
        code = howto_prompt(
            dfname=dfname,
            data_description=description,
            how=how,
            previous_answer=code,
            previous_error=str(e),
        )
    return code


ask_prompt = lambdaprompt.Completion(
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
def ask_from_parts(
    dfname, column_names, data_types, extras, question, index_col_name=None
):
    column_names = from_b64(column_names)
    data_types = from_b64(data_types)
    extras = from_b64(extras)
    description = get_description_from_parts(
        column_names, data_types, extras, index_col_name
    )
    description = pd.json_normalize(description).to_csv(index=False)
    return ask_prompt(dfname=dfname, data_description=description, question=question)


def get_import_modules_from_codestring(code):
    """
    Given a code string, return a list of import module

    eg `from sklearn import linear_model` would return `["sklearn"]`
    eg. `print(3)` would return `[]`
    eg. `import pandas as pd; import matplotlib.pyplot as plt` would return `["pandas", "matplotlib"]`
    """
    # use ast to parse the code
    tree = ast.parse(code)
    # get all the import statements
    import_statements = [node for node in tree.body if isinstance(node, ast.Import)]
    # get all the import from statements
    import_from_statements = [
        node for node in tree.body if isinstance(node, ast.ImportFrom)
    ]
    # get all the module names
    import_modules = []
    for node in import_statements:
        for alias in node.names:
            import_modules.append(alias.name)
    import_modules += [node.module for node in import_from_statements]
    # only take parent module (eg. `matplotlib.pyplot` -> `matplotlib`)
    import_modules = [module.split(".")[0] for module in import_modules]
    return import_modules


def validate_pycode_result(result):
    try:
        modules = get_import_modules_from_codestring(result)
        for module in modules:
            temp = importlib.util.find_spec(module)
            if temp is None:
                logging.warning(
                    f"Module {module} not found, but part of suggestion. May need to pip install..."
                )
    except SyntaxError:
        logging.warning("Syntax error in suggestion -- might not work directly")


@pd.api.extensions.register_dataframe_accessor("sketch")
class SketchHelper:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def howto(self, how, call_display=True):
        result = call_prompt_on_dataframe(self._obj, howto_from_parts, how=how)
        validate_pycode_result(result)
        if not call_display:
            return result
        # output text in a <pre>, also on the side (on top) include a `copy` button that puts it onto clipboard
        uid = uuid.uuid4()
        b64_encoded_result = to_b64(result)
        display(
            HTML(
                f"""<div style="display:flex;flex-direction:row;justify-content:space-between;">
                <pre style="width: 100%; white-space: pre-wrap;" id="{uid}">{result}</pre>
                <button style="height: fit-content;" onclick="navigator.clipboard.writeText(JSON.parse(atob(`{b64_encoded_result}`)))">Copy</button>
                </div>"""
            )
        )

    def ask(self, question, call_display=True):
        result = call_prompt_on_dataframe(self._obj, ask_from_parts, question=question)
        if not call_display:
            return result
        display(HTML(f"""{result}"""))

    def apply(self, prompt_template_string, **kwargs):
        row_limit = int(os.environ.get("SKETCH_ROW_OVERRIDE_LIMIT", "10"))
        if len(self._obj) > row_limit:
            raise RuntimeError(
                f"Too many rows for apply \n (SKETCH_ROW_OVERRIDE_LIMIT: {row_limit}, Actual: {len(self._obj)})"
            )
        new_gpt3_prompt = lambdaprompt.Completion(prompt_template_string)
        named_args = new_gpt3_prompt.get_named_args()
        known_args = set(self._obj.columns) | set(kwargs.keys())
        needed_args = set(named_args)
        if needed_args - known_args:
            raise RuntimeError(
                f"Missing: {needed_args - known_args}\nKnown: {known_args}"
            )

        def apply_func(row):
            row_dict = row.to_dict()
            row_dict.update(kwargs)
            return new_gpt3_prompt(**row_dict)

        return self._obj.apply(apply_func, axis=1)

        # # Async version

        # new_gpt3_prompt = lambdaprompt.AsyncGPT3Prompt(prompt_template_string)
        # named_args = new_gpt3_prompt.get_named_args()
        # known_args = set(self._obj.columns) | set(kwargs.keys())
        # needed_args = set(named_args)
        # if needed_args - known_args:
        #     raise RuntimeError(
        #         f"Missing: {needed_args - known_args}\nKnown: {known_args}"
        #     )

        # ind, vals = [], []
        # for i, row in self._obj.iterrows():
        #     ind.append(i)
        #     row_dict = row.to_dict()
        #     row_dict.update(kwargs)
        #     vals.append(new_gpt3_prompt(**row_dict))

        # # gather the results
        # vals = asyncio.run(asyncio.gather(*vals))

        # return pd.Series(vals, index=ind)
