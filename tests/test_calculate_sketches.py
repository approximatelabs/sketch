import sketch


def test_calculate_sketches(df):
    p = sketch.Portfolio.from_dataframe(df)
    assert len(p.sketchpads) == 4


def test_calculate_sketchpad(df):
    s1 = df["A"]
    sp = sketch.SketchPad.from_series(s1)
    assert len(sp.sketches) > 2
