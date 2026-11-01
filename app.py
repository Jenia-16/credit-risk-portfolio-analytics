import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import roc_curve

st.set_page_config(
    page_title="Credit Risk Portfolio Analytics",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "account_level_validated.csv"
MODEL_DIR = "phase3_outputs"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["origination_date"])

@st.cache_resource
def load_models():
    lr = joblib.load(
        os.path.join(MODEL_DIR, "logistic_regression_pipeline.joblib")
    )
    xgb = joblib.load(
        os.path.join(MODEL_DIR, "xgboost_pipeline.joblib")
    )
    return lr, xgb

@st.cache_data
def load_outputs():
    comparison = pd.read_csv(
        os.path.join(MODEL_DIR, "model_comparison.csv")
    )
    thresholds = pd.read_csv(
        os.path.join(MODEL_DIR, "threshold_analysis.csv")
    )
    importance = pd.read_csv(
        os.path.join(MODEL_DIR, "xgboost_feature_importance.csv")
    )
    with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
        metadata = json.load(f)
    return comparison, thresholds, importance, metadata

df = load_data()
lr_model, xgb_model = load_models()
comparison, thresholds, importance, metadata = load_outputs()

st.title("Credit Risk Portfolio Analytics")
st.caption(
    "Portfolio exploration, retrospective behavioral risk classification, "
    "and explainable risk insights"
)

# Sidebar filters
st.sidebar.header("Portfolio Filters")

filtered = df.copy()

for col in ["loan_type", "region", "employment_type"]:
    if col in filtered.columns:
        options = sorted(filtered[col].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(
            col.replace("_", " ").title(),
            options,
            default=options
        )
        filtered = filtered[
            filtered[col].astype(str).isin(selected)
        ]

if "origination_date" in filtered.columns:
    min_date = filtered["origination_date"].min()
    max_date = filtered["origination_date"].max()

    date_range = st.sidebar.date_input(
        "Origination date range",
        value=(min_date.date(), max_date.date())
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["origination_date"].dt.date.between(
                start_date, end_date
            )
        ]

# Portfolio KPIs
total_accounts = len(filtered)
total_customers = (
    filtered["customer_id"].nunique()
    if "customer_id" in filtered.columns else 0
)
defaults = (
    int(filtered["target_default"].sum())
    if total_accounts else 0
)
default_rate = defaults / total_accounts if total_accounts else 0
total_principal = (
    filtered["loan_amount"].sum()
    if "loan_amount" in filtered.columns else 0
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accounts", f"{total_accounts:,}")
c2.metric("Unique Customers", f"{total_customers:,}")
c3.metric("Default Rate", f"{default_rate:.2%}")
c4.metric("Original Principal", f"${total_principal:,.0f}")

st.divider()

# Portfolio charts
left, right = st.columns(2)

with left:
    st.subheader("Default Rate by Loan Type")
    if "loan_type" in filtered.columns:
        loan_summary = (
            filtered.groupby("loan_type", dropna=False)
            .agg(
                accounts=("account_id", "count"),
                defaults=("target_default", "sum")
            )
            .reset_index()
        )
        loan_summary["default_rate"] = (
            loan_summary["defaults"] / loan_summary["accounts"]
        )
        fig = px.bar(
            loan_summary,
            x="loan_type",
            y="default_rate",
            color="default_rate",
            hover_data=["accounts", "defaults"],
            labels={"default_rate": "Default Rate", "loan_type": "Loan Type"}
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Default Rate by Employment Type")
    if "employment_type" in filtered.columns:
        emp_summary = (
            filtered.groupby("employment_type", dropna=False)
            .agg(
                accounts=("account_id", "count"),
                defaults=("target_default", "sum")
            )
            .reset_index()
        )
        emp_summary["default_rate"] = (
            emp_summary["defaults"] / emp_summary["accounts"]
        )
        fig = px.bar(
            emp_summary,
            x="employment_type",
            y="default_rate",
            color="default_rate",
            hover_data=["accounts", "defaults"],
            labels={
                "default_rate": "Default Rate",
                "employment_type": "Employment Type"
            }
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Default Rate by Income")
    if "annual_income" in filtered.columns:
        income_data = filtered.copy()
        income_data["income_band"] = pd.qcut(
            income_data["annual_income"],
            q=5,
            duplicates="drop"
        )
        income_summary = (
            income_data.groupby("income_band", observed=False)
            .agg(
                accounts=("account_id", "count"),
                defaults=("target_default", "sum")
            )
            .reset_index()
        )
        income_summary["default_rate"] = (
            income_summary["defaults"] / income_summary["accounts"]
        )
        income_summary["income_band"] = (
            income_summary["income_band"].astype(str)
        )
        fig = px.bar(
            income_summary,
            x="income_band",
            y="default_rate",
            hover_data=["accounts", "defaults"],
            labels={
                "income_band": "Income Band",
                "default_rate": "Default Rate"
            }
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Default Rate by Region")
    if "region" in filtered.columns:
        region_summary = (
            filtered.groupby("region", dropna=False)
            .agg(
                accounts=("account_id", "count"),
                defaults=("target_default", "sum")
            )
            .reset_index()
        )
        region_summary["default_rate"] = (
            region_summary["defaults"] / region_summary["accounts"]
        )
        fig = px.bar(
            region_summary,
            x="region",
            y="default_rate",
            hover_data=["accounts", "defaults"],
            labels={"default_rate": "Default Rate", "region": "Region"}
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Model performance
st.header("Model Performance")
st.caption(
    "Metrics shown are from the customer-disjoint test split. "
    "Current model scope is retrospective behavioral classification."
)

st.dataframe(comparison, use_container_width=True, hide_index=True)

left, right = st.columns(2)

with left:
    st.subheader("Threshold Trade-off")
    st.caption("Illustrative review thresholds; not an approved lending cutoff.")
    st.dataframe(
        thresholds.round(3),
        use_container_width=True,
        hide_index=True
    )

with right:
    st.subheader("XGBoost Feature Importance")
    top_features = importance.head(15).sort_values("Importance")
    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        labels={"Importance": "Importance", "Feature": "Feature"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.warning(
    "Model limitation: default timing relative to the six-month repayment "
    "features is not established. Treat predictions as retrospective "
    "behavioral risk signals, not validated origination-time underwriting."
)

st.subheader("Filtered Account Data")
st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)

# -----------------------------
# Account-level Risk Scoring
# -----------------------------
st.divider()
st.header("Account-Level Risk Scoring")

st.caption(
    "Select an account to view its model-estimated behavioral risk "
    "and the strongest contributing factors."
)

account_ids = sorted(df["account_id"].astype(str).unique())

selected_account_id = st.selectbox(
    "Select an account",
    account_ids,
    key="risk_account_selector"
)

selected_account = df[
    df["account_id"].astype(str) == selected_account_id
].iloc[[0]].copy()

model_choice = st.radio(
    "Scoring model",
    ["Logistic Regression", "XGBoost"],
    horizontal=True,
    key="risk_model_choice"
)

selected_model = (
    lr_model if model_choice == "Logistic Regression"
    else xgb_model
)

# Use exactly the feature columns the model was trained on
model_feature_columns = selected_model.named_steps[
    "preprocessor"
].feature_names_in_

account_features = selected_account[
    list(model_feature_columns)
]

risk_probability = selected_model.predict_proba(
    account_features
)[0, 1]

review_threshold = st.slider(
    "Illustrative review threshold",
    min_value=0.10,
    max_value=0.80,
    value=0.40,
    step=0.05,
    key="account_review_threshold"
)

risk_status = (
    "Flag for review"
    if risk_probability >= review_threshold
    else "Below review threshold"
)

# Account summary
c1, c2, c3, c4 = st.columns(4)

c1.metric("Account ID", selected_account_id)
c2.metric("Loan Amount", f"${selected_account['loan_amount'].iloc[0]:,.0f}")
c3.metric("Model Risk Score", f"{risk_probability:.1%}")
c4.metric("Review Status", risk_status)

st.caption(
    "This is a retrospective behavioral risk score, not a validated "
    "origination-time underwriting probability or an automatic lending decision."
)

# Show selected account details
with st.expander("View account details"):
    display_columns = [
        col for col in [
            "account_id",
            "loan_type",
            "loan_purpose",
            "loan_amount",
            "interest_rate_pct",
            "tenure_months",
            "age",
            "employment_type",
            "annual_income",
            "region",
            "target_default"
        ]
        if col in selected_account.columns
    ]

    st.dataframe(
        selected_account[display_columns].T.rename(
            columns={selected_account.index[0]: "Value"}
        ),
        use_container_width=True
    )

# Individual explanation for XGBoost
if model_choice == "XGBoost":
    st.subheader("Why did the model assign this score?")

    try:
        import shap

        preprocessor = xgb_model.named_steps["preprocessor"]
        classifier = xgb_model.named_steps["classifier"]

        transformed_account = preprocessor.transform(account_features)
        transformed_feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        individual_shap_values = explainer.shap_values(
            transformed_account
        )

        values = np.array(individual_shap_values)

        # Handle binary-class SHAP output formats
        if values.ndim == 3:
            values = values[0, :, 1]
        else:
            values = values.reshape(-1)

        explanation_df = pd.DataFrame({
            "Feature": transformed_feature_names,
            "SHAP Value": values
        })

        explanation_df["Impact"] = np.where(
            explanation_df["SHAP Value"] > 0,
            "Pushes toward higher model score",
            "Pushes toward lower model score"
        )

        explanation_df["Absolute Impact"] = (
            explanation_df["SHAP Value"].abs()
        )

        explanation_df = explanation_df.sort_values(
            "Absolute Impact",
            ascending=False
        ).head(12)

        fig = px.bar(
            explanation_df.sort_values("SHAP Value"),
            x="SHAP Value",
            y="Feature",
            color="Impact",
            orientation="h",
            title="Top contributing features for this account"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            explanation_df[
                ["Feature", "SHAP Value", "Impact"]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(f"Could not generate this account's SHAP explanation: {e}")

else:
    st.info(
        "For feature-level SHAP explanations, select XGBoost. "
        "Logistic Regression scoring is available above."
    )
    # -----------------------------
# Anomaly Detection & Review
# -----------------------------
st.divider()
st.header("Anomaly Detection & Human Review")

st.caption(
    "Unusual account patterns are flagged for investigation. "
    "An anomaly is not proof of fraud, default, or misconduct."
)

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Use selected numeric features available in the dataset.
# Exclude IDs, target, collections outcomes, and other post-outcome fields.
anomaly_candidate_features = [
    "loan_amount",
    "interest_rate_pct",
    "tenure_months",
    "age",
    "annual_income",
    "years_employed",
    "years_at_residence",
    "bureau_prior_line_count",
    "bureau_prior_sanctioned_total",
    "bureau_prior_sanctioned_mean",
    "bureau_prior_credit_type_count",
    "bureau_prior_lender_count",
    "missed_payments_6m",
    "late_payments_6m",
    "avg_dpd_6m",
    "max_dpd_6m",
    "total_emi_due_6m",
    "total_emi_paid_6m",
    "total_shortfall_6m",
    "avg_paid_fraction_6m",
    "balance_after_month6"
]

anomaly_features = [
    c for c in anomaly_candidate_features
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
]

if len(anomaly_features) < 2:
    st.warning("Not enough numeric features available for anomaly detection.")
else:
    st.write("Features used:", ", ".join(anomaly_features))

    contamination = st.slider(
        "Expected share of accounts to flag",
        min_value=0.01,
        max_value=0.10,
        value=0.03,
        step=0.01,
        help=(
            "This controls how many accounts are flagged for review. "
            "It is not an estimated fraud or default rate."
        ),
        key="anomaly_contamination"
    )

    # Fit anomaly detector on the portfolio
    anomaly_input = df[anomaly_features].copy()

    imputer = SimpleImputer(strategy="median")
    anomaly_matrix = imputer.fit_transform(anomaly_input)

    anomaly_model = IsolationForest(
        n_estimators=250,
        contamination=contamination,
        random_state=42
    )

    anomaly_labels = anomaly_model.fit_predict(anomaly_matrix)
    anomaly_scores = anomaly_model.decision_function(anomaly_matrix)

    anomaly_results = df[
        ["account_id", "customer_id", "loan_type", "loan_amount",
         "region", "employment_type"]
    ].copy()

    anomaly_results["Anomaly Score"] = anomaly_scores
    anomaly_results["Flagged"] = anomaly_labels == -1

    # Lower Isolation Forest decision scores indicate more unusual records.
    anomaly_results = anomaly_results.sort_values(
        "Anomaly Score",
        ascending=True
    )

    flagged_results = anomaly_results[
        anomaly_results["Flagged"]
    ].copy()

    # Generate simple feature-level reasons for each flagged account.
    feature_medians = anomaly_input.median()
    feature_iqrs = (
        anomaly_input.quantile(0.75) -
        anomaly_input.quantile(0.25)
    ).replace(0, np.nan)

    def anomaly_reason(row):
        deviations = (
            (anomaly_input.loc[row.name] - feature_medians)
            / feature_iqrs
        ).replace([np.inf, -np.inf], np.nan).dropna()

        if deviations.empty:
            return "Unusual combination of account characteristics"

        top_features = deviations.abs().sort_values(
            ascending=False
        ).head(3).index.tolist()

        return "Unusual values: " + ", ".join(
            feature.replace("_", " ") for feature in top_features
        )

    flagged_results["Reason for Flag"] = [
        anomaly_reason(row)
        for _, row in flagged_results.iterrows()
    ]

    # Store review statuses in Streamlit session state
    if "anomaly_review_status" not in st.session_state:
        st.session_state["anomaly_review_status"] = {}

    flagged_results["Review Status"] = flagged_results[
        "account_id"
    ].astype(str).map(
        st.session_state["anomaly_review_status"]
    ).fillna("Pending")

    # Summary metrics
    a1, a2, a3 = st.columns(3)

    a1.metric("Accounts Analyzed", f"{len(anomaly_results):,}")
    a2.metric("Accounts Flagged", f"{len(flagged_results):,}")
    a3.metric(
        "Flag Rate",
        f"{len(flagged_results) / len(anomaly_results):.1%}"
        if len(anomaly_results) else "0.0%"
    )

    st.subheader("Flagged Accounts")

    if flagged_results.empty:
        st.info("No accounts were flagged at this setting.")
    else:
        st.dataframe(
            flagged_results[
                [
                    "account_id",
                    "loan_type",
                    "loan_amount",
                    "region",
                    "Anomaly Score",
                    "Reason for Flag",
                    "Review Status"
                ]
            ].round(4),
            use_container_width=True,
            hide_index=True
        )

        # Human-review controls
        st.subheader("Review an Account")

        review_account = st.selectbox(
            "Select flagged account",
            flagged_results["account_id"].astype(str).tolist(),
            key="anomaly_review_account"
        )

        current_status = st.session_state[
            "anomaly_review_status"
        ].get(review_account, "Pending")

        new_status = st.selectbox(
            "Review status",
            ["Pending", "Reviewed", "Escalated"],
            index=["Pending", "Reviewed", "Escalated"].index(
                current_status
            ),
            key="anomaly_status_selector"
        )

        if st.button("Save Review Status"):
            st.session_state[
                "anomaly_review_status"
            ][review_account] = new_status

            st.success(
                f"Updated {review_account} to: {new_status}"
            )
            st.rerun()

        # Download current review queue
        updated_queue = flagged_results.copy()
        updated_queue["Review Status"] = updated_queue[
            "account_id"
        ].astype(str).map(
            st.session_state["anomaly_review_status"]
        ).fillna("Pending")

        st.download_button(
            "Download Anomaly Review Queue (CSV)",
            data=updated_queue.to_csv(index=False),
            file_name="anomaly_review_queue.csv",
            mime="text/csv"
        )

    st.info(
        "Review statuses are currently held in the active Streamlit session. "
        "They may reset when the app or runtime restarts. "
        "The downloaded CSV preserves the current review queue."
    )
    # -----------------------------
# Ask the Portfolio
# -----------------------------
st.divider()
st.header("Ask the Portfolio")

st.caption(
    "Ask questions about portfolio metrics and segments. "
    "Answers are calculated from the loaded dataset; unsupported questions "
    "are declined rather than guessed."
)

import re

def portfolio_answer(question, data):
    import re
    import pandas as pd

    q = question.lower().strip()
    df = data.copy()

    if df.empty:
        return "There are no accounts in the current filtered portfolio."

    # Ensure target is numeric/binary
    df["target_default"] = pd.to_numeric(
        df["target_default"], errors="coerce"
    )
    df = df.dropna(subset=["target_default"])

    if df.empty:
        return "Default labels are unavailable for the current portfolio."

    total_accounts = len(df)
    total_defaults = int(df["target_default"].sum())
    overall_rate = df["target_default"].mean()
    total_principal = pd.to_numeric(
        df["loan_amount"], errors="coerce"
    ).sum()

    # Helper: calculate group default rates safely
    def grouped_default_rate(column, label):
        if column not in df.columns:
            return f"I can't calculate default rate by {label}: that field is unavailable."

        grouped = (
            df.groupby(column, dropna=False)
              .agg(
                  accounts=("target_default", "size"),
                  defaults=("target_default", "sum"),
                  default_rate=("target_default", "mean")
              )
              .reset_index()
        )

        grouped = grouped[grouped["accounts"] > 0]

        if grouped.empty:
            return f"No usable {label} groups are available."

        # Highest rate; if tied, report all tied groups
        max_rate = grouped["default_rate"].max()
        winners = grouped[
            grouped["default_rate"].round(10) == round(max_rate, 10)
        ]

        winner_text = ", ".join(
            f"{row[column]} ({row['default_rate']:.2%}, "
            f"{int(row['defaults']):,}/{int(row['accounts']):,} accounts)"
            for _, row in winners.iterrows()
        )

        table = grouped.sort_values(
            "default_rate", ascending=False
        )

        breakdown = "\n".join(
            f"- {row[column]}: {row['default_rate']:.2%} "
            f"({int(row['defaults']):,}/{int(row['accounts']):,})"
            for _, row in table.iterrows()
        )

        return (
            f"The {label} with the highest default rate is **{winner_text}**.\n\n"
            f"**Breakdown:**\n{breakdown}"
        )

    # 1. Specific "highest default rate by region" intent FIRST
    region_highest_patterns = [
        "which region has the highest default rate",
        "what region has the highest default rate",
        "which region has highest default rate",
        "region with the highest default rate",
        "highest default rate by region",
        "highest default rate region",
        "worst region by default rate",
        "region with the most defaults rate",
    ]

    if any(pattern in q for pattern in region_highest_patterns):
        return grouped_default_rate("region", "region")

    # Flexible phrasing for regional ranking
    if (
        "region" in q
        and ("highest" in q or "max" in q or "worst" in q or "rank" in q)
        and ("default rate" in q or "default" in q)
    ):
        return grouped_default_rate("region", "region")

    # 2. Other "highest default rate by ..." questions
    if (
        ("highest" in q or "max" in q or "worst" in q)
        and ("default rate" in q or "default" in q)
    ):
        if "loan type" in q or "loan category" in q:
            return grouped_default_rate("loan_type", "loan type")
        if "employment" in q or "job" in q:
            return grouped_default_rate("employment_type", "employment type")
        if "income" in q:
            # Create income bands for a useful breakdown
            if "annual_income" not in df.columns:
                return "Annual income is unavailable for the current portfolio."

            income = pd.to_numeric(df["annual_income"], errors="coerce")
            valid = income.notna()

            if valid.sum() < 2:
                return "There isn't enough valid income data to compare income bands."

            df_income = df.loc[valid].copy()
            df_income["income_band"] = pd.qcut(
                pd.to_numeric(df_income["annual_income"]),
                q=5,
                duplicates="drop"
            )

            grouped = (
                df_income.groupby("income_band", observed=False)
                .agg(
                    accounts=("target_default", "size"),
                    defaults=("target_default", "sum"),
                    default_rate=("target_default", "mean")
                )
                .reset_index()
            )

            grouped = grouped[grouped["accounts"] > 0]
            if grouped.empty:
                return "No usable income bands are available."

            top = grouped.sort_values("default_rate", ascending=False).iloc[0]

            breakdown = "\n".join(
                f"- {row['income_band']}: {row['default_rate']:.2%} "
                f"({int(row['defaults']):,}/{int(row['accounts']):,})"
                for _, row in grouped.sort_values(
                    "default_rate", ascending=False
                ).iterrows()
            )

            return (
                f"The income band with the highest default rate is "
                f"**{top['income_band']}** at **{top['default_rate']:.2%}** "
                f"({int(top['defaults']):,}/{int(top['accounts']):,} accounts).\n\n"
                f"**Breakdown:**\n{breakdown}"
            )

    # 3. Exact total original principal intent
    if (
        "original principal" in q
        or "total principal" in q
        or "principal amount" in q
        or ("principal" in q and ("total" in q or "amount" in q))
    ):
        return (
            f"The total original principal is **${total_principal:,.2f}** "
            f"across **{total_accounts:,} accounts**. "
            "This is original loan principal, not current exposure, EAD, "
            "or realized loss."
        )

    # 4. Account/customer counts
    if "unique customer" in q or "distinct customer" in q:
        if "customer_id" not in df.columns:
            return "Customer IDs are unavailable for the current portfolio."
        return f"There are **{df['customer_id'].nunique():,} unique customers**."

    if (
        "account count" in q
        or "number of accounts" in q
        or "how many accounts" in q
        or "total accounts" in q
        or "portfolio size" in q
    ):
        return f"There are **{total_accounts:,} accounts** in the current portfolio."

    # 5. Overall default rate ONLY when no segment/ranking was requested
    asks_overall_rate = (
        "overall default rate" in q
        or "portfolio default rate" in q
        or q in {
            "default rate",
            "what is the default rate",
            "what's the default rate",
            "what is our default rate",
        }
    )

    if asks_overall_rate:
        return (
            f"The default rate is **{overall_rate:.2%}** "
            f"({total_defaults:,} defaults out of {total_accounts:,} accounts)."
        )

    # 6. Default rate by specified segment
    if "default rate" in q or "default rates" in q:
        if "loan type" in q or "loan category" in q:
            return grouped_default_rate("loan_type", "loan type")
        if "employment" in q or "job" in q:
            return grouped_default_rate("employment_type", "employment type")
        if "region" in q:
            return grouped_default_rate("region", "region")

    # 7. Unsupported question: don't substitute a different metric
    return (
        "I can't answer that reliably from the currently supported portfolio "
        "calculations. Try asking about account counts, total original principal, "
        "overall default rate, or default rates by loan type, employment type, "
        "region, or income band."
    )
    
    # -----------------------------
# Gemini: grounded answer layer
# -----------------------------
import streamlit as st
from google import genai

def gemini_portfolio_answer(question, data):
    # First calculate the answer deterministically from the filtered data
    calculated_answer = portfolio_answer(question, data)

    unsupported_message = (
        "I can't answer that reliably from the currently supported portfolio "
        "calculations."
    )

    # Do not let the LLM invent an answer for unsupported questions
    if calculated_answer.startswith(unsupported_message):
        return calculated_answer

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are an assistant for a credit risk portfolio analytics dashboard. Answer the users question using ONLY the verified calculation below. Do not add statistics, facts, explanations, or recommendations that are not supported by that calculation. Preserve all numbers and percentages exactly. If the calculation does not answer the question, say that the information is not available from the current portfolio calculations. Do not make lending decisions or imply that correlations are causal. Keep the answer concise and clear. User question: {question} Verified portfolio calculation: {calculated_answer} """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        if response.text:
            return response.text.strip()

        return calculated_answer

    except Exception as e:
        # If Gemini is unavailable, the dashboard still returns the verified result
        return (
            calculated_answer
            + "\n\n_Gemini is temporarily unavailable; showing the "
              "verified calculated answer instead._"
        )
# Question input and Gemini answer display
question = st.text_input(
    "Ask a question about the portfolio",
    placeholder="e.g. Which region has the highest default rate?"
)

if st.button("Ask", key="ask_portfolio_button"):
    if question.strip():
        with st.spinner("Analyzing the portfolio..."):
            answer = gemini_portfolio_answer(question, filtered)

        st.markdown(answer)
    else:
        st.warning("Please enter a question first.")
