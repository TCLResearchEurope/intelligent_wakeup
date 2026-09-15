import pandas as pd

from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "pollResults.csv"

NON_INFERIORITY_MARGIN = -0.30


DIMENSIONS = [
    "Speech naturalness",
    "Interaction naturalness",
    "Scenario plausibility",
    "Conversational coherence",
    "Overall conversation naturalness",
]


PRIMARY_DIMENSION = "Overall conversation naturalness"


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading data...")

df = pd.read_csv(CSV_FILE)

# ============================================================
# BASIC DATA CHECKS
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECKS")
print("=" * 70)

print(f"Total rating rows: {len(df):,}")
print(f"Participants: {df['participant_id'].nunique():,}")
print(f"Unique clips: {df['stimulus_id'].nunique():,}")

print("\nRatings per participant:")
print(df.groupby("participant_id").size().describe())

print("\nRatings per clip:")
print(df.groupby("stimulus_id").size().describe())

print("\nMissing scores:")
print(df["score"].isna().sum())

print("\nScore distribution:")
print(df["score"].value_counts().sort_index())

print(df[:10])


# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)


descriptive = (
    df
    .groupby(["dimension", "corpus"])["score"]
    .agg(
        n="count",
        mean="mean",
        std="std",
        median="median",
        min="min",
        max="max",
    )
    .reset_index()
)


print(
    descriptive.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# ============================================================
# SIMPLE MEAN DIFFERENCES
# ============================================================

means = (
    df
    .groupby(["dimension", "corpus"])["score"]
    .mean()
    .unstack()
)


means["difference_proposed_minus_natural"] = (
    means["proposed"] - means["natural"]
)


print("\n" + "=" * 70)
print("RAW MEAN DIFFERENCES")
print("=" * 70)

print(
    means.to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

# ============================================================
# LMM FUNCTION
# ============================================================

def fit_lmm(data, dimension):
    """
    Fit:

        rating ~ corpus + (1 | annotator) + (1 | clip)

    using statsmodels' MixedLM.
    """

    d = data[data["dimension"] == dimension].copy()

    d["corpus"] = pd.Categorical(
        d["corpus"],
        categories=["natural", "proposed"]
    )

    formula = "score ~ C(corpus)"

    if "scenario" in d.columns:
        if d["scenario"].notna().any():
            formula += " + C(scenario)"

    if "duration" in d.columns:
        if d["duration"].notna().any():
            formula += " + duration"

    print("\n")
    print("-" * 70)
    print(f"LMM: {dimension}")
    print("-" * 70)
    print(f"Formula: {formula}")

    model = smf.mixedlm(
        formula=formula,
        data=d,
        groups=d["participant_id"],
        vc_formula={
            "clip": "0 + C(stimulus_id)"
        },
        re_formula="1",
    )

    result = model.fit(
        reml=False,
        method="lbfgs"
    )

    return result


# ============================================================
# PRIMARY ANALYSIS — Q5
# ============================================================

print("\n" + "=" * 70)
print("PRIMARY ANALYSIS")
print("=" * 70)

primary_result = fit_lmm(
    df,
    PRIMARY_DIMENSION
)


print(primary_result.summary())

parameter_name = "C(corpus)[T.proposed]"

if parameter_name not in primary_result.params.index:

    print("\nCould not automatically find the corpus coefficient.")
    print("Available parameters:")
    print(primary_result.params)

    raise SystemExit


delta = primary_result.params[parameter_name]
se = primary_result.bse[parameter_name]

z_90 = stats.norm.ppf(0.95)

ci_lower = delta - z_90 * se
ci_upper = delta + z_90 * se


print("\n" + "=" * 70)
print("PRIMARY NON-INFERIORITY RESULT")
print("=" * 70)

print(f"Estimated difference (proposed - natural): {delta:.3f}")
print(f"Standard error: {se:.3f}")
print(
    f"Two-sided 90% CI: "
    f"[{ci_lower:.3f}, {ci_upper:.3f}]"
)
print(
    f"One-sided 95% lower bound: "
    f"{ci_lower:.3f}"
)
print(
    f"Non-inferiority margin: "
    f"{NON_INFERIORITY_MARGIN:.2f}"
)


if ci_lower > NON_INFERIORITY_MARGIN:

    primary_conclusion = (
        "NON-INFERIORITY SUPPORTED"
    )

else:

    primary_conclusion = (
        "NON-INFERIORITY NOT DEMONSTRATED"
    )


print(f"\nPRIMARY CONCLUSION: {primary_conclusion}")


# ============================================================
# CHECK FOR EVIDENCE OF A DIFFERENCE
# ============================================================

p_value_primary = primary_result.pvalues[parameter_name]

print("\nDifference from zero:")
print(f"p = {p_value_primary:.6f}")


if p_value_primary < 0.05:

    if delta < 0:
        print(
            "There is evidence that the proposed corpus "
            "has lower ratings than the natural corpus."
        )
    else:
        print(
            "There is evidence that the proposed corpus "
            "has higher ratings than the natural corpus."
        )

else:

    print(
        "There is no statistically significant evidence "
        "of a difference from zero."
    )


# ============================================================
# SECONDARY OUTCOMES Q1-Q4
# ============================================================

print("\n" + "=" * 70)
print("SECONDARY OUTCOMES")
print("=" * 70)


secondary_results = []


for dimension in DIMENSIONS:

    if dimension == PRIMARY_DIMENSION:
        continue

    try:

        result = fit_lmm(df, dimension)

        if parameter_name not in result.params.index:
            print(
                f"Could not extract corpus effect for {dimension}"
            )
            continue

        effect = result.params[parameter_name]
        se = result.bse[parameter_name]
        p = result.pvalues[parameter_name]

        secondary_results.append({
            "dimension": dimension,
            "effect": effect,
            "SE": se,
            "p_raw": p,
        })

    except Exception as e:

        print(
            f"\nCould not fit model for {dimension}:"
        )
        print(e)


secondary_results = pd.DataFrame(
    secondary_results
)


# ============================================================
# HOLM CORRECTION
# ============================================================

if len(secondary_results) > 0:

    rejected, p_corrected, _, _ = multipletests(
        secondary_results["p_raw"],
        alpha=0.05,
        method="holm"
    )

    secondary_results["p_holm"] = p_corrected
    secondary_results["significant_holm"] = rejected

    print("\nHolm-corrected secondary results:")

    print(
        secondary_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# APPROXIMATE OBSERVED PRECISION
# ============================================================

print("\n" + "=" * 70)
print("PRECISION")
print("=" * 70)

print(
    f"Observed SE for primary corpus effect: {se:.4f}"
)

print(
    f"Approximate 95% CI half-width: "
    f"{1.96 * se:.3f}"
)

print(
    "\nThe study design expected approximately:"
)

print(
    "SE ≈ 0.115 points"
)

print(
    "95% CI half-width ≈ 0.23 points"
)


# ============================================================
# COMPARE OBSERVED SE WITH DESIGN EXPECTATION
# ============================================================

expected_se = 0.115

ratio = se / expected_se

print(
    f"\nObserved SE / expected SE: {ratio:.2f}"
)

if ratio < 1:

    print(
        "The observed estimate is more precise than "
        "the original design approximation."
    )

elif ratio > 1:

    print(
        "The observed estimate is less precise than "
        "the original design approximation."
    )

else:

    print(
        "The observed precision is approximately equal "
        "to the original design approximation."
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL STUDY CONCLUSION")
print("=" * 70)

print(
    f"""
Primary outcome:
{PRIMARY_DIMENSION}

Estimated corpus difference:
    Δ = proposed - natural = {delta:.3f}

Two-sided 90% confidence interval:
    [{ci_lower:.3f}, {ci_upper:.3f}]

Non-inferiority margin:
    {NON_INFERIORITY_MARGIN:.2f}

Result:
    {primary_conclusion}

Interpretation:
"""
)


if ci_lower > NON_INFERIORITY_MARGIN:

    print(
        "The lower confidence bound is above the "
        "predefined non-inferiority margin of -0.30. "
        "Therefore, the proposed corpus is supported "
        "as non-inferior to the natural corpus with "
        "respect to overall conversational naturalness."
    )

else:

    print(
        "The lower confidence bound does not exceed "
        "the predefined non-inferiority margin of -0.30. "
        "Therefore, non-inferiority of the proposed corpus "
        "has not been demonstrated."
    )


print(
    """
Important:
Failure to demonstrate non-inferiority does NOT by itself
demonstrate that the proposed corpus is inferior.

Likewise, a non-significant difference from zero does NOT
by itself establish non-inferiority. The non-inferiority
confidence-bound criterion above is the relevant primary
decision rule.
"""
)


# ============================================================
# SAVE RESULTS
# ============================================================

descriptive.to_csv(
    "descriptive_statistics.csv",
    index=False
)

df.to_csv(
    "analysis_dataset.csv",
    index=False
)

if len(secondary_results) > 0:

    secondary_results.to_csv(
        "secondary_lmm_results.csv",
        index=False
    )


print("\nSaved:")
print("  analysis_dataset.csv")
print("  descriptive_statistics.csv")

if len(secondary_results) > 0:
    print("  secondary_lmm_results.csv")

print("\nAnalysis complete.")
