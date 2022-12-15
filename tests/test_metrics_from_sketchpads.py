import sketch


def test_calculate_unary_metrics(df):
    p = sketch.Portfolio.from_dataframe(df)
    for s in p.sketchpads.values():
        metrics = s.get_metrics()


def test_calculate_cross_metrics(df):
    p = sketch.Portfolio.from_dataframe(df)
    for s1 in p.sketchpads.values():
        for s2 in p.sketchpads.values():
            metrics = s1.get_cross_metrics(s2)
