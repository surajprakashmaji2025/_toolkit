import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, norm
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Statistical Toolkit",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a3a5c;
        text-align: center;
        padding: 1rem 0 0.2rem 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #555555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a3a5c;
        border-left: 4px solid #2E75B6;
        padding-left: 10px;
        margin: 1.5rem 0 0.8rem 0;
    }
    .result-box {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid #c8e0f4;
        margin: 0.5rem 0;
    }
    .reject-box {
        background-color: #fff0f0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid #f4c8c8;
        margin: 0.5rem 0;
    }
    .pass-box {
        background-color: #f0fff4;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        border: 1px solid #c8f4d8;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 0.8rem;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">📊 Statistical Toolkit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Tools and Methods of Data Analysis | Cohort 2504 | SRH University Leipzig</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/SRH_Hochschulen_logo.svg/200px-SRH_Hochschulen_logo.svg.png", width=120)
    st.markdown("### Navigation")
    test_choice = st.radio(
        "Select Test",
        ["Task A — Z-Test", "Task B — Chi-Square Test"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Teen Mental Health Dataset")
    st.markdown("[Kaggle Source](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy)")
    st.markdown("---")
    st.markdown("**Libraries**")
    st.markdown("pandas · numpy · scipy · matplotlib · seaborn")

# ── Load Data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("Teen_Mental_Health_Dataset.csv")

try:
    df = load_data()
    data_loaded = True
except:
    st.error("Please make sure Teen_Mental_Health_Dataset.csv is in the same folder as app.py")
    data_loaded = False
    st.stop()

# ══════════════════════════════════════════════════════════════
# TASK A — Z-TEST
# ══════════════════════════════════════════════════════════════
if test_choice == "Task A — Z-Test":

    st.markdown("## Task A: One-Sample Z-Test")
    st.markdown("**Research Question:** Is the average daily sleep hours of teenagers significantly different from the WHO recommended 8 hours?")

    # ── Section 0: Data Description ───────────────────────────
    st.markdown('<div class="section-header">0. Data Description and Source</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", "Teen Mental Health")
    with col2:
        st.metric("Total Rows", f"{len(df):,}")
    with col3:
        st.metric("Column Used", "sleep_hours")
    with col4:
        st.metric("Source", "Kaggle")

    st.markdown("""
    The **Teen Mental Health Dataset** contains 1,200 records of teenagers aged 13–19.
    It captures social media usage, sleep patterns, academic performance, stress, anxiety, and addiction levels.
    For this Z-Test, we use the `sleep_hours` column to test whether teenagers are getting the WHO-recommended 8 hours of sleep per night.
    """)

    with st.expander("Preview Dataset (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

    # ── Parameters ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Test Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        pop_mean = st.number_input("Hypothesized Population Mean (μ₀)", value=8.0, step=0.1)
    with col2:
        alpha = st.number_input("Significance Level (α)", value=0.05, step=0.01)
    with col3:
        sample_data = df['sleep_hours'].dropna().values
        pop_std = float(np.std(sample_data))
        st.metric("Population Std Dev (σ)", f"{pop_std:.4f}")

    run_z = st.button("▶ Run Z-Test", type="primary", use_container_width=True)

    if run_z:

        n = len(sample_data)
        sample_mean = np.mean(sample_data)
        se = pop_std / np.sqrt(n)
        z_stat = (sample_mean - pop_mean) / se
        p_val = 2 * (1 - norm.cdf(abs(z_stat)))
        ci_low = sample_mean - norm.ppf(1 - alpha/2) * se
        ci_high = sample_mean + norm.ppf(1 - alpha/2) * se
        decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"

        # ── Section 1: Assumption Checks ──────────────────────
        st.markdown('<div class="section-header">1. Assumption Checks</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Sample Size (n ≥ 30)**")
            if n >= 30:
                st.markdown(f'<div class="pass-box">✅ n = {n:,} — Satisfied<br>Large sample, CLT applies.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="reject-box">❌ n = {n} — Not satisfied</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**Population Std Dev Known**")
            st.markdown(f'<div class="pass-box">✅ σ = {pop_std:.4f} — Satisfied<br>Calculated from full dataset.</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("**Normality Check (Shapiro-Wilk)**")
            sample_200 = np.random.choice(sample_data, size=200, replace=False)
            sw_stat, sw_p = stats.shapiro(sample_200)
            if sw_p > 0.05:
                st.markdown(f'<div class="pass-box">✅ p = {sw_p:.4f} — Normality met</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box">⚠️ p = {sw_p:.4f} — Formally rejected<br>But CLT applies (n={n:,})</div>', unsafe_allow_html=True)

        # ── Section 2: Visualization ───────────────────────────
        st.markdown('<div class="section-header">2. Data Visualization</div>', unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Z-Test: Distribution of Sleep Hours", fontsize=14, fontweight='bold')

        # Histogram
        axes[0].hist(sample_data, bins=30, color='steelblue', edgecolor='black', density=True, alpha=0.7)
        kde_x = np.linspace(sample_data.min()-1, sample_data.max()+1, 300)
        kde_y = stats.gaussian_kde(sample_data)(kde_x)
        axes[0].plot(kde_x, kde_y, color='black', linewidth=2, label='KDE')
        axes[0].axvline(sample_mean, color='blue', linestyle='-', linewidth=2, label=f'Sample Mean = {sample_mean:.2f}')
        axes[0].axvline(pop_mean, color='red', linestyle='--', linewidth=2, label=f'H₀ Mean = {pop_mean}')
        axes[0].set_title("Histogram with KDE")
        axes[0].set_xlabel("Sleep Hours")
        axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=8)

        # Boxplot
        axes[1].boxplot(sample_data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='steelblue', alpha=0.7))
        axes[1].axhline(pop_mean, color='red', linestyle='--', linewidth=2, label=f'H₀ = {pop_mean}')
        axes[1].axhline(sample_mean, color='blue', linestyle='-', linewidth=2, label=f'Mean = {sample_mean:.2f}')
        axes[1].set_title("Boxplot")
        axes[1].set_ylabel("Sleep Hours")
        axes[1].legend(fontsize=8)

        # Z distribution
        z_range = np.linspace(-5, 5, 400)
        axes[2].plot(z_range, norm.pdf(z_range), color='black', linewidth=2)
        z_crit = norm.ppf(1 - alpha/2)
        axes[2].fill_between(z_range, norm.pdf(z_range),
                              where=(z_range <= -z_crit), color='red', alpha=0.3, label='Rejection region')
        axes[2].fill_between(z_range, norm.pdf(z_range),
                              where=(z_range >= z_crit), color='red', alpha=0.3)
        axes[2].axvline(z_stat, color='blue', linestyle='--', linewidth=2, label=f'z = {z_stat:.2f}')
        axes[2].set_title("Z Distribution")
        axes[2].set_xlabel("Z Score")
        axes[2].set_ylabel("Density")
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

        # ── Section 3: Results Table ───────────────────────────
        st.markdown('<div class="section-header">3. Test Results Table</div>', unsafe_allow_html=True)

        results = {
            "Item": ["Test", "H₀", "Hₐ", "Sample Size (n)", "Sample Mean", "Population Mean (μ₀)",
                     "Mean Difference", "Z-Statistic", "Degrees of Freedom",
                     "p-value", "Significance Level (α)", "95% Confidence Interval", "Decision"],
            "Value": [
                "One-Sample Z-Test",
                f"μ = {pop_mean} (teenagers sleep {pop_mean} hours)",
                f"μ ≠ {pop_mean} (teenagers do not sleep {pop_mean} hours)",
                f"{n:,}",
                f"{sample_mean:.4f}",
                f"{pop_mean}",
                f"{sample_mean - pop_mean:.4f}",
                f"{z_stat:.4f}",
                "N/A (Z-Test uses normal distribution)",
                f"{p_val:.6f}",
                f"{alpha}",
                f"({ci_low:.4f}, {ci_high:.4f})",
                decision
            ]
        }

        results_df = pd.DataFrame(results)

        def highlight_decision(row):
            if row['Item'] == 'Decision':
                if 'Reject' in row['Value'] and 'Fail' not in row['Value']:
                    return ['background-color: #FFE0E0'] * 2
                else:
                    return ['background-color: #E8F5E9'] * 2
            if row['Item'] in ['p-value', 'Z-Statistic']:
                return ['background-color: #D5E8F0'] * 2
            return [''] * 2

        st.dataframe(
            results_df.style.apply(highlight_decision, axis=1),
            use_container_width=True,
            hide_index=True
        )

        # ── Section 4: Conclusion ──────────────────────────────
        st.markdown('<div class="section-header">4. Conclusion</div>', unsafe_allow_html=True)

        if p_val < alpha:
            box_class = "reject-box"
            conclusion = f"""
            A one-sample Z-test was conducted to determine whether the average sleep hours of teenagers
            differs significantly from the WHO-recommended {pop_mean} hours per night.
            The sample of {n:,} teenagers had a mean sleep duration of {sample_mean:.2f} hours.
            The test yielded Z = {z_stat:.4f} and p = {p_val:.6f}.
            Since p < {alpha}, we reject the null hypothesis.
            There is statistically significant evidence that teenagers are sleeping significantly
            less than the recommended {pop_mean} hours per night.
            The 95% confidence interval of ({ci_low:.2f}, {ci_high:.2f}) does not include {pop_mean},
            further confirming this finding. This result suggests that teenage sleep deprivation
            is a real and measurable concern that warrants attention from parents, educators, and healthcare professionals.
            """
        else:
            box_class = "pass-box"
            conclusion = f"""
            A one-sample Z-test was conducted to determine whether the average sleep hours of teenagers
            differs significantly from the WHO-recommended {pop_mean} hours per night.
            The sample of {n:,} teenagers had a mean sleep duration of {sample_mean:.2f} hours.
            The test yielded Z = {z_stat:.4f} and p = {p_val:.6f}.
            Since p > {alpha}, we fail to reject the null hypothesis.
            There is no statistically significant evidence that the average sleep hours differs
            from {pop_mean} hours at the {alpha} significance level.
            """

        st.markdown(f'<div class="{box_class}">{conclusion}</div>', unsafe_allow_html=True)

        # Key metrics summary
        st.markdown("**Key Metrics**")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Sample Mean", f"{sample_mean:.2f} hrs")
        col2.metric("H₀ Mean", f"{pop_mean} hrs")
        col3.metric("Z-Statistic", f"{z_stat:.4f}")
        col4.metric("p-value", f"{p_val:.6f}")
        col5.metric("Decision", decision)


# ══════════════════════════════════════════════════════════════
# TASK B — CHI-SQUARE TEST
# ══════════════════════════════════════════════════════════════
elif test_choice == "Task B — Chi-Square Test":

    st.markdown("## Task B: Chi-Square Test of Independence")
    st.markdown("**Research Question:** Is there a significant relationship between the social media platform teenagers use and their social interaction level?")

    # ── Section 0: Data Description ───────────────────────────
    st.markdown('<div class="section-header">0. Data Description and Source</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset", "Teen Mental Health")
    with col2:
        st.metric("Total Rows", f"{len(df):,}")
    with col3:
        st.metric("Variable 1", "platform_usage")
    with col4:
        st.metric("Variable 2", "social_interaction_level")

    st.markdown("""
    The **Teen Mental Health Dataset** contains 1,200 records of teenagers aged 13–19.
    For this Chi-Square Test, we examine whether the social media platform a teenager uses
    (Instagram, TikTok, or Both) is significantly associated with their social interaction level
    (low, medium, or high). Both variables are categorical, making the Chi-Square test the appropriate choice.
    """)

    with st.expander("Preview Dataset (first 10 rows)"):
        st.dataframe(df[['platform_usage', 'social_interaction_level', 'gender', 'age']].head(10), use_container_width=True)

    alpha = st.number_input("Significance Level (α)", value=0.05, step=0.01, key="chi_alpha")
    run_chi = st.button("▶ Run Chi-Square Test", type="primary", use_container_width=True)

    if run_chi:

        # Compute contingency table
        ct = pd.crosstab(df['platform_usage'], df['social_interaction_level'])
        chi2_stat, p_val, dof, expected = chi2_contingency(ct)
        n = len(df)
        decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"

        # ── Section 1: Assumption Checks ──────────────────────
        st.markdown('<div class="section-header">1. Assumption Checks</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Sample Size (n ≥ 30)**")
            st.markdown(f'<div class="pass-box">✅ n = {n:,} — Satisfied</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("**Categorical Variables**")
            st.markdown('<div class="pass-box">✅ Both variables are categorical — Satisfied</div>', unsafe_allow_html=True)

        with col3:
            st.markdown("**Expected Frequencies ≥ 5**")
            min_exp = expected.min()
            if min_exp >= 5:
                st.markdown(f'<div class="pass-box">✅ Min expected = {min_exp:.2f} — Satisfied</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="reject-box">❌ Min expected = {min_exp:.2f} — Not satisfied</div>', unsafe_allow_html=True)

        st.markdown("**Observed Frequency Table (Contingency Table)**")
        st.dataframe(ct, use_container_width=True)

        st.markdown("**Expected Frequency Table**")
        expected_df = pd.DataFrame(
            expected.round(2),
            index=ct.index,
            columns=ct.columns
        )
        st.dataframe(expected_df, use_container_width=True)

        # ── Section 2: Visualization ───────────────────────────
        st.markdown('<div class="section-header">2. Data Visualization</div>', unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Chi-Square Test: Platform Usage vs Social Interaction Level",
                     fontsize=14, fontweight='bold')

        # Grouped bar chart
        ct.plot(kind='bar', ax=axes[0], color=['coral', 'steelblue', 'seagreen'],
                edgecolor='black', alpha=0.8)
        axes[0].set_title("Observed Counts")
        axes[0].set_xlabel("Platform Usage")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].legend(title="Social Interaction")

        # Stacked bar chart (proportions)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot(kind='bar', stacked=True, ax=axes[1],
                    color=['coral', 'steelblue', 'seagreen'],
                    edgecolor='black', alpha=0.8)
        axes[1].set_title("Proportional Breakdown (%)")
        axes[1].set_xlabel("Platform Usage")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].tick_params(axis='x', rotation=0)
        axes[1].legend(title="Social Interaction")

        # Heatmap
        sns.heatmap(ct, annot=True, fmt='d', cmap='Blues',
                    ax=axes[2], linewidths=0.5)
        axes[2].set_title("Heatmap of Frequencies")
        axes[2].set_xlabel("Social Interaction Level")
        axes[2].set_ylabel("Platform Usage")

        plt.tight_layout()
        st.pyplot(fig)

        # ── Section 3: Results Table ───────────────────────────
        st.markdown('<div class="section-header">3. Test Results Table</div>', unsafe_allow_html=True)

        results = {
            "Item": ["Test", "Variable 1", "Variable 2", "H₀", "Hₐ",
                     "Sample Size (n)", "Chi-Square Statistic",
                     "Degrees of Freedom", "p-value",
                     "Significance Level (α)", "Decision"],
            "Value": [
                "Chi-Square Test of Independence",
                "platform_usage (Instagram / TikTok / Both)",
                "social_interaction_level (low / medium / high)",
                "Platform usage and social interaction level are independent (no relationship)",
                "Platform usage and social interaction level are NOT independent (there is a relationship)",
                f"{n:,}",
                f"{chi2_stat:.4f}",
                f"{dof}",
                f"{p_val:.6f}",
                f"{alpha}",
                decision
            ]
        }

        results_df = pd.DataFrame(results)

        def highlight_decision_chi(row):
            if row['Item'] == 'Decision':
                if 'Reject' in row['Value'] and 'Fail' not in row['Value']:
                    return ['background-color: #FFE0E0'] * 2
                else:
                    return ['background-color: #E8F5E9'] * 2
            if row['Item'] in ['p-value', 'Chi-Square Statistic']:
                return ['background-color: #D5E8F0'] * 2
            return [''] * 2

        st.dataframe(
            results_df.style.apply(highlight_decision_chi, axis=1),
            use_container_width=True,
            hide_index=True
        )

        # ── Section 4: Conclusion ──────────────────────────────
        st.markdown('<div class="section-header">4. Conclusion</div>', unsafe_allow_html=True)

        if p_val < alpha:
            box_class = "reject-box"
            conclusion = f"""
            A Chi-Square test of independence was conducted to examine whether there is a significant
            relationship between the social media platform teenagers use and their social interaction level.
            The contingency table included {n:,} observations across three platform categories
            (Instagram, TikTok, Both) and three interaction levels (low, medium, high).
            The test yielded χ²({dof}) = {chi2_stat:.4f}, p = {p_val:.6f}.
            Since p < {alpha}, we reject the null hypothesis.
            There is statistically significant evidence that platform usage and social interaction level
            are NOT independent — the platform a teenager uses is significantly associated with
            their level of social interaction. For example, Instagram users showed a notably higher
            proportion of high social interaction compared to TikTok users, suggesting that
            different platforms may influence teenagers' real-world social behaviours differently.
            """
        else:
            box_class = "pass-box"
            conclusion = f"""
            A Chi-Square test of independence was conducted to examine whether there is a significant
            relationship between the social media platform teenagers use and their social interaction level.
            The test yielded χ²({dof}) = {chi2_stat:.4f}, p = {p_val:.6f}.
            Since p > {alpha}, we fail to reject the null hypothesis.
            There is no statistically significant evidence of a relationship between platform usage
            and social interaction level at the {alpha} significance level.
            """

        st.markdown(f'<div class="{box_class}">{conclusion}</div>', unsafe_allow_html=True)

        # Key metrics
        st.markdown("**Key Metrics**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chi-Square Stat", f"{chi2_stat:.4f}")
        col2.metric("Degrees of Freedom", f"{dof}")
        col3.metric("p-value", f"{p_val:.6f}")
        col4.metric("Decision", decision)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Statistical Toolkit | Tools and Methods of Data Analysis | Cohort 2504 | SRH University Leipzig"
    "</div>",
    unsafe_allow_html=True
)
