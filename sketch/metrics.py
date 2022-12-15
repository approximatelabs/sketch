import datasketches
import numpy as np


def strings_from_sketchpad_sketches(sketchpad):
    # FI and VO are the two
    output = ""
    ds = sketchpad.get_sketchdata_by_name("DS_FI")
    # consider showing the counts of frequent items?? Might be useful information.
    output += " ".join(
        [
            x[0]
            for x in ds.get_frequent_items(
                datasketches.frequent_items_error_type.NO_FALSE_POSITIVES
            )
        ]
    )
    output += "\n"
    output += " ".join(
        [
            x[0]
            for x in ds.get_frequent_items(
                datasketches.frequent_items_error_type.NO_FALSE_NEGATIVES
            )
        ]
    )
    output += "\n"
    ds = sketchpad.get_sketchdata_by_name("DS_VO")
    output += " ".join([x[0] for x in ds.get_samples()])
    return output


def unary_metrics(sketchpad):
    # get metrics for a single sketchpad
    # return a vector of metrics
    metrics = {}

    metrics["rows"] = sketchpad.get_sketchdata_by_name("Rows")
    metrics["count"] = sketchpad.get_sketchdata_by_name("Count")

    ds = sketchpad.get_sketchdata_by_name("DS_HLL")

    metrics["hll_lower_bound_2"] = ds.get_lower_bound(2)
    metrics["hll_upper_bound_2"] = ds.get_upper_bound(2)
    metrics["hll_estimate"] = ds.get_estimate()

    ds = sketchpad.get_sketchdata_by_name("DS_CPC")
    metrics["cpc_lower_bound_2"] = ds.get_lower_bound(2)
    metrics["cpc_upper_bound_2"] = ds.get_upper_bound(2)
    metrics["cpc_estimate"] = ds.get_estimate()

    ds = sketchpad.get_sketchdata_by_name("DS_THETA")
    metrics["theta_lower_bound_2"] = ds.get_lower_bound(2)
    metrics["theta_upper_bound_2"] = ds.get_upper_bound(2)
    metrics["theta_estimate"] = ds.get_estimate()

    ds = sketchpad.get_sketchdata_by_name("DS_FI")
    # likely can't use these, as they are more... values of data than metrics
    # metrics["fi_no_false_pos"] = ds.get_frequent_items(datasketches.frequent_items_error_type.NO_FALSE_POSITIVES)
    # metrics["fi_no_false_neg"] = ds.get_frequent_items(datasketches.frequent_items_error_type.NO_FALSE_NEGATIVES)

    ds = sketchpad.get_sketchdata_by_name("DS_KLL")
    # pts = ds.get_quantiles([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    metrics["kll_quantile_0.01"] = ds.get_quantile(0.01)
    metrics["kll_quantile_0.1"] = ds.get_quantile(0.1)
    metrics["kll_quantile_0.25"] = ds.get_quantile(0.25)
    metrics["kll_quantile_0.5"] = ds.get_quantile(0.5)
    metrics["kll_quantile_0.75"] = ds.get_quantile(0.75)
    metrics["kll_quantile_0.9"] = ds.get_quantile(0.9)
    metrics["kll_quantile_0.99"] = ds.get_quantile(0.99)

    ds = sketchpad.get_sketchdata_by_name("DS_Quantiles")
    metrics["quantiles_quantile_0.01"] = ds.get_quantile(0.01)
    metrics["quantiles_quantile_0.1"] = ds.get_quantile(0.1)
    metrics["quantiles_quantile_0.25"] = ds.get_quantile(0.25)
    metrics["quantiles_quantile_0.5"] = ds.get_quantile(0.5)
    metrics["quantiles_quantile_0.75"] = ds.get_quantile(0.75)
    metrics["quantiles_quantile_0.9"] = ds.get_quantile(0.9)
    metrics["quantiles_quantile_0.99"] = ds.get_quantile(0.99)

    ds = sketchpad.get_sketchdata_by_name("DS_REQ")
    metrics["req_min_value"] = ds.get_min_value()
    metrics["req_max_value"] = ds.get_max_value()
    # not sure, should i include quantiles or specific "rank" get values?

    # VO Sketch has failed
    # ds = wow.get_sketchdata_by_name("DS_VO")
    # print("=VO=".ljust(12, " "), ds.to_string(True))

    ds = sketchpad.get_sketchdata_by_name("UnicodeMatches")
    metrics.update({f"unicode_{k}": v for k, v in ds.items()})

    return metrics


def max_delta(x1, y1, x2, y2):
    f1 = np.interp(np.concatenate([x1, x2]), x2, y2)
    f2 = np.interp(np.concatenate([x1, x2]), x1, y1)
    return np.max(np.abs(f1 - f2))


def get_CDF(s, N=100):
    yvals = [x / N for x in range(N + 1)]
    xvals = s.get_quantiles(yvals)
    return xvals, yvals


def ks_estimate(s1, s2):
    # Need to do a smarter job of handling nulls or something
    x1, y1 = get_CDF(s1)
    x2, y2 = get_CDF(s2)
    return max_delta(x1, y1, x2, y2)


def binary_metrics(sketchpad1, sketchpad2):
    metrics = {}

    ds1 = sketchpad1.get_sketchdata_by_name("DS_THETA")
    ds2 = sketchpad2.get_sketchdata_by_name("DS_THETA")

    lower, estimate, upper = datasketches.theta_jaccard_similarity.jaccard(ds1, ds2)
    metrics["theta_jaccard_lower_bound"] = lower
    metrics["theta_jaccard_upper_bound"] = upper
    metrics["theta_jaccard_estimate"] = estimate
    metrics["theta_exactly_equal"] = int(
        datasketches.theta_jaccard_similarity.exactly_equal(ds1, ds2)
    )
    theta_1_not_2 = datasketches.theta_a_not_b().compute(ds1, ds2)
    metrics["theta_1_not_2"] = theta_1_not_2.get_estimate()
    theta_2_not_1 = datasketches.theta_a_not_b().compute(ds2, ds1)
    metrics["theta_2_not_1"] = theta_2_not_1.get_estimate()
    intersect = datasketches.theta_intersection()
    intersect.update(ds1)
    intersect.update(ds2)
    metrics["theta_intersection_estimate"] = intersect.get_result().get_estimate()

    # Share same frequent items
    ds1 = sketchpad1.get_sketchdata_by_name("DS_FI")
    ds2 = sketchpad2.get_sketchdata_by_name("DS_FI")

    fi1 = ds1.get_frequent_items(
        datasketches.frequent_items_error_type.NO_FALSE_POSITIVES
    )
    fi2 = ds2.get_frequent_items(
        datasketches.frequent_items_error_type.NO_FALSE_POSITIVES
    )
    fi1 = [x[0] for x in fi1]
    fi2 = [x[0] for x in fi2]
    metrics["fi_intersection"] = len(set(fi1).intersection(set(fi2)))
    metrics["fi_1_not_2"] = len(set(fi1).difference(set(fi2)))
    metrics["fi_2_not_1"] = len(set(fi2).difference(set(fi1)))

    # KS test
    ds1 = sketchpad1.get_sketchdata_by_name("DS_KLL")
    ds2 = sketchpad2.get_sketchdata_by_name("DS_KLL")

    metrics["ks_test_0.9"] = int(datasketches.ks_test(ds1, ds2, 0.9))
    metrics["ks_test_0.5"] = int(datasketches.ks_test(ds1, ds2, 0.5))
    metrics["ks_test_0.1"] = int(datasketches.ks_test(ds1, ds2, 0.1))
    metrics["ks_test_0.01"] = int(datasketches.ks_test(ds1, ds2, 0.01))
    metrics["ks_test_0.001"] = int(datasketches.ks_test(ds1, ds2, 0.001))
    # if metrics["ks_test_0.5"]:
    #     metrics["kll_ks_score"] = ks_estimate(ds1, ds2)
    # else:
    #     metrics["kll_ks_score"] = 1.0
    return metrics
