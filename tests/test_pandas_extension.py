import pandas as pd

import sketch  # noqa


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        pass


def test_sketch(mocker):
    mocker.patch("requests.get", return_value=FakeResponse("Hello World"))
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["a", "b", "c"],
            "c": [None, 4.1, 3],
            "d": ["010222", "010222", "010222"],
            "e": [[1, 2, 3], [3, 1], []],
        }
    )
    result = df.sketch.ask("What is in column e?", call_display=False)
