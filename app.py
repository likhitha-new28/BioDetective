import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from biodetective.analysis.batch_effects import analyze_batch_pca_association
from biodetective.analysis.confounding import analyze_confounding
from biodetective.analysis.expression_qc import calculate_gene_variance, run_expression_qc
from biodetective.analysis.label_consistency import LabelConsistencyConfig, analyze_label_consistency
from biodetective.analysis.metadata_mapper import (
    SEMANTIC_ROLES,
    create_mapping_approval,
    ranked_columns_for_role,
    suggest_metadata_roles,
)
from biodetective.analysis.metadata_qc import run_metadata_qc
from biodetective.analysis.outliers import OutlierConfig, analyze_outliers
from biodetective.analysis.pca import PCAConfig, run_pca
from biodetective.analysis.similarity import analyze_sample_similarity
from biodetective.core.exceptions import DataLoadError
from biodetective.core.pipeline import PipelineConfig, run_biodetective_pipeline
from biodetective.io.loaders import load_biodataset
from biodetective.io.validators import validate_dataset
from biodetective.reporting import (
    explain_finding,
    generate_analysis_json,
    generate_findings_csv,
    generate_html_report,
    generate_sample_scores_csv,
)
from biodetective.reporting.exports import sorted_findings


def render_finding_details(finding) -> None:
    """Render one Finding through the centralized explanation template."""
    explanation = explain_finding(finding)
    st.markdown("**Observation**")
    st.write(explanation.observation)
    st.markdown("**Evidence**")
    st.json(explanation.evidence)
    if finding.sample_ids:
        st.write("Affected samples:", ", ".join(finding.sample_ids))
    st.markdown("**Interpretation**")
    st.write(explanation.interpretation)
    st.write("Possible explanations include:")
    for possibility in explanation.possible_explanations:
        st.markdown(f"- {possibility}")
    st.markdown("**Recommendation**")
    st.write(explanation.recommendation)


def render_metadata_mapping(metadata: pd.DataFrame):
    """Collect explicit researcher approval for every semantic role mapping."""
    st.markdown("### Metadata Role Mapping")
    st.caption("Review or override each deterministic suggestion, then approve every role. “Not mapped” is a valid choice.")
    suggestions = suggest_metadata_roles(metadata)
    selectable_columns = [str(column) for column in metadata.columns if str(column) != "sample_id"]
    mappings: dict[str, str | None] = {}
    approved_roles: list[str] = []
    mapping_columns = st.columns(2)

    for index, role in enumerate(SEMANTIC_ROLES):
        candidates = ranked_columns_for_role(suggestions, role)
        suggested_column = candidates[0].column if candidates else None
        options = [None, *selectable_columns]
        default_index = options.index(suggested_column) if suggested_column in options else 0
        with mapping_columns[index % 2]:
            selected = st.selectbox(
                f"{role.title()} column",
                options,
                index=default_index,
                format_func=lambda value: "Not mapped" if value is None else value,
                key=f"metadata_mapping_{role}",
            )
            mappings[role] = selected
            selected_suggestion = next(
                (candidate for candidate in candidates if candidate.column == selected),
                None,
            )
            if selected_suggestion is not None:
                reasons = "; ".join(selected_suggestion.reasons)
                st.caption(f"Heuristic score: {selected_suggestion.score:.2f}. {reasons}")
            elif selected is not None:
                st.caption("Researcher override; this column was not the highest-ranked heuristic suggestion.")
            approved = st.checkbox(
                f"Approve {role} mapping",
                key=f"approve_metadata_mapping_{role}_{selected}",
            )
            if approved:
                approved_roles.append(role)

    with st.expander("View all ranked mapping suggestions"):
        suggestion_records = [
            {
                "column": suggestion.column,
                "role": suggestion.role,
                "score": suggestion.score,
                "reasons": "; ".join(suggestion.reasons),
            }
            for role in SEMANTIC_ROLES
            for suggestion in ranked_columns_for_role(suggestions, role)
        ]
        if suggestion_records:
            st.dataframe(pd.DataFrame(suggestion_records), use_container_width=True, hide_index=True)
        else:
            st.info("No metadata-role candidates met the deterministic heuristic threshold.")

    approval = create_mapping_approval(metadata, mappings, approved_roles)
    if not approval.fully_approved:
        remaining = [role for role in SEMANTIC_ROLES if role not in approval.approved_roles]
        st.info("Approve every mapping before analysis: " + ", ".join(remaining) + ".")
    return approval


def render_overview_dashboard(result) -> None:
    """Render overview metrics using only completed pipeline results."""
    st.markdown("## Overview Dashboard")
    health_score = f"{result.dataset_health.score:.1f}/100" if result.dataset_health is not None else "Unavailable"
    metadata_cells = result.dataset.metadata.size
    missing_values = int(result.dataset.metadata.isna().sum().sum())
    metadata_completeness = 100.0 * (1 - missing_values / metadata_cells) if metadata_cells else 100.0
    suspicious_samples = sum(score.risk in {"High", "Critical"} for score in result.sample_scores)
    severity_counts = {
        severity: sum(finding.severity == severity for finding in result.findings)
        for severity in ("critical", "high", "medium")
    }

    first_metrics = st.columns(4)
    first_metrics[0].metric("Dataset Health Score", health_score)
    first_metrics[1].metric("Samples", result.dataset.n_samples)
    first_metrics[2].metric("Genes", result.dataset.n_features)
    first_metrics[3].metric("Metadata completeness", f"{metadata_completeness:.1f}%")
    second_metrics = st.columns(4)
    second_metrics[0].metric("High/Critical review samples", suspicious_samples)
    second_metrics[1].metric("Critical findings", severity_counts["critical"])
    second_metrics[2].metric("High findings", severity_counts["high"])
    second_metrics[3].metric("Moderate findings", severity_counts["medium"])

    st.markdown("### Top findings")
    top_findings = sorted_findings(result.findings)[:10]
    if top_findings:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "severity": finding.severity.title(),
                        "category": finding.category,
                        "observation": explain_finding(finding).observation,
                        "evidence": str(explain_finding(finding).evidence),
                        "interpretation": explain_finding(finding).interpretation,
                        "samples": ", ".join(finding.sample_ids),
                        "recommendation": explain_finding(finding).recommendation,
                    }
                    for finding in top_findings
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No findings were produced by the completed analyses.")


def render_sample_explorer(result) -> None:
    """Render existing per-sample evidence without running new analyses."""
    st.markdown("## Sample Explorer")
    selected_sample = st.selectbox("Sample ID", result.dataset.sample_ids, key="sample_explorer_id")
    score = next((item for item in result.sample_scores if item.sample_id == selected_sample), None)
    sample_findings = [finding for finding in result.findings if selected_sample in finding.sample_ids]

    metadata = result.dataset.metadata.copy()
    if "sample_id" in metadata.columns:
        metadata_rows = metadata.loc[metadata["sample_id"].astype(str).eq(selected_sample)]
        st.markdown("### Metadata")
        st.dataframe(metadata_rows, use_container_width=True, hide_index=True)

    st.markdown("### Suspicion score")
    if score is None:
        st.info("No suspicion score is available for this sample.")
    else:
        score_metrics = st.columns(2)
        score_metrics[0].metric("Score", f"{score.score:.1f}/100")
        score_metrics[1].metric("Risk", score.risk)
        breakdown = pd.DataFrame(
            [
                {"evidence source": source.replace("_", " ").title(), "contribution": contribution}
                for source, contribution in score.contributions.items()
            ]
        )
        st.markdown("#### Score breakdown")
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.markdown("### PCA coordinates")
    pca_module = result.modules.get("pca")
    if pca_module is not None and pca_module.status == "completed" and selected_sample in pca_module.result.coordinates.index:
        coordinates = pca_module.result.coordinates.loc[[selected_sample]].reset_index()
        st.dataframe(coordinates, use_container_width=True, hide_index=True)
    else:
        st.info("PCA coordinates are unavailable for this sample.")

    st.markdown("### Nearest/highly correlated samples")
    similarity_module = result.modules.get("similarity")
    if similarity_module is not None and similarity_module.status == "completed":
        correlation_matrix = similarity_module.result[0]
        if selected_sample in correlation_matrix.index:
            correlations = correlation_matrix.loc[selected_sample].drop(labels=[selected_sample], errors="ignore")
            nearest = correlations.dropna().sort_values(ascending=False).head(10)
            similarity_table = nearest.rename("correlation").rename_axis("sample_id").reset_index()
            flagged_samples = {
                sample_id
                for finding in sample_findings
                if finding.category == "sample_similarity"
                for sample_id in finding.sample_ids
                if sample_id != selected_sample
            }
            similarity_table["flagged_similarity"] = similarity_table["sample_id"].isin(flagged_samples)
            st.dataframe(similarity_table, use_container_width=True, hide_index=True)
        else:
            st.info("Correlation results are unavailable for this sample.")
    else:
        st.info("Sample similarity analysis was not completed.")

    st.markdown("### Findings affecting this sample")
    if sample_findings:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "severity": finding.severity.title(),
                        "category": finding.category,
                        "observation": explain_finding(finding).observation,
                        "evidence": str(explain_finding(finding).evidence),
                        "interpretation": explain_finding(finding).interpretation,
                        "recommendation": explain_finding(finding).recommendation,
                    }
                    for finding in sorted_findings(sample_findings)
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        for finding in sorted_findings(sample_findings):
            with st.expander(f"[{finding.severity.upper()}] {finding.message}"):
                render_finding_details(finding)
        st.markdown("### Recommended actions")
        for recommendation in dict.fromkeys(explain_finding(finding).recommendation for finding in sample_findings):
            st.markdown(f"- {recommendation}")
    else:
        st.success("No findings directly affect this sample.")
        st.markdown("### Recommended actions")
        st.write("No finding-specific action is currently recommended.")


def render_report_downloads(result) -> None:
    """Render downloads generated from analysis results, never raw expression data."""
    st.markdown("## Download Reports")
    download_columns = st.columns(4)
    download_columns[0].download_button(
        "Download findings.csv",
        generate_findings_csv(result),
        file_name="findings.csv",
        mime="text/csv",
    )
    download_columns[1].download_button(
        "Download sample_scores.csv",
        generate_sample_scores_csv(result),
        file_name="sample_scores.csv",
        mime="text/csv",
    )
    download_columns[2].download_button(
        "Download analysis.json",
        generate_analysis_json(result),
        file_name="analysis.json",
        mime="application/json",
    )
    download_columns[3].download_button(
        "Download HTML report",
        generate_html_report(result),
        file_name="biodetective_report.html",
        mime="text/html",
    )


st.set_page_config(page_title="BioDetective", page_icon="🧬", layout="wide")

st.title("BioDetective")
st.subheader("Gene-Expression Data Quality Evidence & Review")

expression_file = st.file_uploader("Upload expression CSV", type="csv")
metadata_file = st.file_uploader("Upload metadata CSV", type="csv")

if expression_file is not None and metadata_file is not None:
    try:
        dataset = load_biodataset(expression_file, metadata_file)
        validation = validate_dataset(dataset)

        if validation.is_valid:
            st.success("Dataset validated. Ready for analysis.")
        else:
            st.error("The dataset could not be validated.")
            for issue in validation.errors:
                st.error(issue.message)

        missing_metadata = int(dataset.metadata.isna().sum().sum())
        metric_columns = st.columns(4)
        metric_columns[0].metric("Samples", dataset.n_samples)
        metric_columns[1].metric("Genes/features", dataset.n_features)
        metric_columns[2].metric("Metadata columns", len(dataset.metadata_columns))
        metric_columns[3].metric("Missing metadata values", missing_metadata)

        st.markdown("### Expression preview")
        st.dataframe(dataset.expression.head(), use_container_width=True)

        st.markdown("### Metadata preview")
        st.dataframe(dataset.metadata.head(), use_container_width=True)

        st.markdown("## Unified Analysis")
        mapping_approval = render_metadata_mapping(dataset.metadata)
        if st.button(
            "Run BioDetective Analysis",
            type="primary",
            disabled=not validation.is_valid or not mapping_approval.fully_approved,
        ):
            mappings = mapping_approval.mappings
            pipeline_config = PipelineConfig(
                label_column=mappings["condition"],
                sex_column=mappings["sex"],
                batch_column=mappings["batch"],
                biological_column=mappings["condition"] if mappings["batch"] is not None else None,
                technical_column=mappings["batch"] if mappings["condition"] is not None else None,
                metadata_mappings=mappings,
            )
            progress_bar = st.progress(0.0, text="Starting BioDetective analysis...")

            def update_pipeline_progress(module_name: str, completed: int, total: int) -> None:
                label = module_name.replace("_", " ").title()
                progress_bar.progress(completed / total, text=f"{label}: {completed}/{total} modules")

            pipeline_result = run_biodetective_pipeline(
                dataset,
                config=pipeline_config,
                progress_callback=update_pipeline_progress,
            )
            st.session_state["biodetective_pipeline_result"] = pipeline_result
            progress_bar.progress(1.0, text="BioDetective analysis complete.")

        pipeline_result = st.session_state.get("biodetective_pipeline_result")
        if pipeline_result is not None:
            st.success("BioDetective analysis complete. Results are stored for this session.")
            module_statuses = pd.DataFrame(
                [
                    {
                        "module": name.replace("_", " ").title(),
                        "status": outcome.status.title(),
                        "message": outcome.message or "",
                    }
                    for name, outcome in pipeline_result.modules.items()
                ]
            )
            st.dataframe(module_statuses, use_container_width=True, hide_index=True)
            render_overview_dashboard(pipeline_result)
            render_sample_explorer(pipeline_result)
            render_report_downloads(pipeline_result)

        st.markdown("## Metadata Audit")
        findings = run_metadata_qc(dataset.metadata)

        severity_order = ["critical", "high", "medium", "low"]
        severity_counts = {severity: sum(finding.severity == severity for finding in findings) for severity in severity_order}
        audit_metrics = st.columns(5)
        audit_metrics[0].metric("Total findings", len(findings))
        for metric, severity in zip(audit_metrics[1:], severity_order):
            metric.metric(severity.title(), severity_counts[severity])

        chart_columns = st.columns(2)
        missing_counts = dataset.metadata.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        with chart_columns[0]:
            st.markdown("### Missing values")
            if missing_counts.empty:
                st.info("No missing metadata values detected.")
            else:
                missing_frame = missing_counts.rename_axis("column").reset_index(name="missing_count")
                missing_figure = px.bar(
                    missing_frame,
                    x="column",
                    y="missing_count",
                    labels={"column": "Metadata column", "missing_count": "Missing values"},
                )
                st.plotly_chart(missing_figure, use_container_width=True)

        categorical_columns = [
            column
            for column in dataset.metadata.columns
            if column != "sample_id"
            and (
                isinstance(dataset.metadata[column].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(dataset.metadata[column].dtype)
                or pd.api.types.is_string_dtype(dataset.metadata[column].dtype)
            )
        ]
        with chart_columns[1]:
            st.markdown("### Categorical distributions")
            if categorical_columns:
                selected_column = st.selectbox("Metadata column", categorical_columns)
                distribution = (
                    dataset.metadata[selected_column]
                    .fillna("<missing>")
                    .astype(str)
                    .value_counts()
                    .rename_axis("value")
                    .reset_index(name="count")
                )
                distribution_figure = px.bar(
                    distribution,
                    x="value",
                    y="count",
                    labels={"value": selected_column, "count": "Samples"},
                )
                st.plotly_chart(distribution_figure, use_container_width=True)
            else:
                st.info("No categorical metadata columns are available.")

        st.markdown("### Findings")
        if not findings:
            st.success("No metadata quality findings detected.")
        else:
            findings_table = pd.DataFrame(
                [
                    {
                        "severity": finding.severity,
                        "category": finding.category,
                        "code": finding.code,
                        "column": finding.column or "",
                        "affected_samples": len(finding.sample_ids),
                        "message": finding.message,
                    }
                    for finding in findings
                ]
            )
            st.dataframe(findings_table, use_container_width=True, hide_index=True)

            for index, finding in enumerate(findings, start=1):
                title = f"{index}. [{finding.severity.upper()}] {finding.message}"
                with st.expander(title):
                    render_finding_details(finding)

        st.markdown("## Expression QC")
        expression_findings, variance_result, sample_statistics = run_expression_qc(dataset.expression)

        st.markdown("### Invalid-value and variance findings")
        if expression_findings:
            expression_findings_table = pd.DataFrame(
                [
                    {
                        "severity": finding.severity,
                        "code": finding.code,
                        "affected_samples": len(finding.sample_ids),
                        "message": finding.message,
                    }
                    for finding in expression_findings
                ]
            )
            st.dataframe(expression_findings_table, use_container_width=True, hide_index=True)
            for finding in expression_findings:
                with st.expander(f"[{finding.severity.upper()}] {finding.message}"):
                    render_finding_details(finding)
        else:
            st.success("No expression quality findings detected.")

        st.markdown("### Sample summary statistics")
        st.dataframe(sample_statistics, use_container_width=True)

        use_log_scale = st.checkbox("Visualize using log2(x + 1)")
        visual_expression = dataset.expression.replace([np.inf, -np.inf], np.nan).copy()
        if use_log_scale:
            if visual_expression.lt(-1).any().any():
                st.warning("Values below -1 cannot be displayed using log2(x + 1) and are omitted from plots.")
            with np.errstate(divide="ignore", invalid="ignore"):
                visual_expression = np.log2(visual_expression + 1)

        plot_columns = st.columns(2)
        with plot_columns[0]:
            st.markdown("### Missing values by sample")
            expression_missing = (
                dataset.expression.isna()
                .sum(axis=0)
                .rename_axis("sample_id")
                .reset_index(name="missing_count")
            )
            missing_expression_figure = px.bar(
                expression_missing,
                x="sample_id",
                y="missing_count",
                labels={"sample_id": "Sample", "missing_count": "Missing expression values"},
            )
            st.plotly_chart(missing_expression_figure, use_container_width=True)

        with plot_columns[1]:
            st.markdown("### Gene variance distribution")
            visual_variances = calculate_gene_variance(visual_expression).dropna()
            variance_frame = visual_variances.rename("variance").reset_index()
            variance_figure = px.histogram(
                variance_frame,
                x="variance",
                nbins=40,
                labels={"variance": "Gene variance"},
            )
            st.plotly_chart(variance_figure, use_container_width=True)

        st.markdown("### Sample expression distributions")
        expression_for_plot = visual_expression.copy()
        expression_for_plot.index.name = expression_for_plot.index.name or "gene_id"
        index_column = expression_for_plot.index.name
        expression_long = expression_for_plot.reset_index().melt(
            id_vars=index_column,
            var_name="sample_id",
            value_name="expression",
        )
        distribution_figure = px.box(
            expression_long,
            x="sample_id",
            y="expression",
            points=False,
            labels={
                "sample_id": "Sample",
                "expression": "log2(expression + 1)" if use_log_scale else "Expression",
            },
        )
        st.plotly_chart(distribution_figure, use_container_width=True)

        st.markdown("## Sample Similarity")
        correlation_method = st.selectbox("Correlation method", ["Pearson", "Spearman"])
        correlation_matrix, similarity_findings = analyze_sample_similarity(
            dataset.expression,
            metadata=dataset.metadata,
            method=correlation_method.casefold(),
        )

        st.markdown("### Correlation heatmap")
        heatmap = px.imshow(
            correlation_matrix,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            labels={"x": "Sample", "y": "Sample", "color": "Correlation"},
            aspect="auto",
        )
        st.plotly_chart(heatmap, use_container_width=True)

        st.markdown("### Potentially highly similar sample pairs")
        if similarity_findings:
            pair_table = pd.DataFrame(
                [
                    {
                        "sample_1": finding.evidence["sample_1"],
                        "sample_2": finding.evidence["sample_2"],
                        "correlation": finding.evidence["correlation"],
                        "level": finding.evidence["similarity_level"],
                        "metadata_conflicts": ", ".join(finding.evidence["metadata_differences"].keys()) or "None",
                    }
                    for finding in similarity_findings
                ]
            )
            st.dataframe(pair_table, use_container_width=True, hide_index=True)
            for finding in similarity_findings:
                with st.expander(f"[{finding.severity.upper()}] {finding.message}"):
                    render_finding_details(finding)
        else:
            st.info("No sample pairs met the similarity thresholds.")

        st.markdown("### Pairwise correlation distribution")
        upper_rows, upper_columns = np.triu_indices(len(correlation_matrix), k=1)
        pairwise_values = correlation_matrix.to_numpy(dtype=float)[upper_rows, upper_columns]
        pairwise_values = pairwise_values[np.isfinite(pairwise_values)]
        pairwise_frame = pd.DataFrame({"correlation": pairwise_values})
        pairwise_figure = px.histogram(
            pairwise_frame,
            x="correlation",
            nbins=40,
            labels={"correlation": f"{correlation_method} correlation"},
        )
        st.plotly_chart(pairwise_figure, use_container_width=True)

        st.markdown("## PCA")
        pca_controls = st.columns(3)
        remove_zero_variance = pca_controls[0].checkbox("Remove zero-variance genes", value=True)
        use_top_variable_genes = pca_controls[1].checkbox("Select top variable genes", value=False)
        top_variable_genes = pca_controls[1].number_input(
            "Number of top variable genes",
            min_value=1,
            max_value=max(dataset.n_features, 1),
            value=min(500, max(dataset.n_features, 1)),
            disabled=not use_top_variable_genes,
        )
        selected_gene_limit = int(top_variable_genes) if use_top_variable_genes else dataset.n_features
        maximum_pca_components = min(10, dataset.n_samples, selected_gene_limit)

        if maximum_pca_components < 2:
            st.warning("PCA visualization requires at least two samples and two usable genes.")
        else:
            component_count = pca_controls[2].number_input(
                "PCA components",
                min_value=2,
                max_value=maximum_pca_components,
                value=min(3, maximum_pca_components),
            )
            try:
                pca_result = run_pca(
                    dataset.expression,
                    PCAConfig(
                        remove_zero_variance_genes=remove_zero_variance,
                        top_variable_genes=int(top_variable_genes) if use_top_variable_genes else None,
                        n_components=int(component_count),
                    ),
                )

                available_pairs = [
                    ("PC1", "PC2"),
                    ("PC1", "PC3"),
                    ("PC2", "PC3"),
                ]
                available_pairs = [
                    pair
                    for pair in available_pairs
                    if pair[0] in pca_result.coordinates.columns and pair[1] in pca_result.coordinates.columns
                ]
                pca_plot_controls = st.columns(2)
                selected_pair_label = pca_plot_controls[0].selectbox(
                    "PCA axes",
                    [f"{first} vs {second}" for first, second in available_pairs],
                )
                selected_pair = available_pairs[
                    [f"{first} vs {second}" for first, second in available_pairs].index(selected_pair_label)
                ]
                color_options = ["None", *categorical_columns]
                color_column = pca_plot_controls[1].selectbox("Color samples by", color_options)

                pca_plot_data = pca_result.coordinates.reset_index()
                if color_column != "None":
                    metadata_for_plot = dataset.metadata.drop_duplicates("sample_id", keep="first").copy()
                    metadata_for_plot["sample_id"] = metadata_for_plot["sample_id"].astype(str)
                    pca_plot_data = pca_plot_data.merge(
                        metadata_for_plot[["sample_id", color_column]],
                        on="sample_id",
                        how="left",
                    )

                x_component, y_component = selected_pair
                x_variance = pca_result.explained_variance[x_component] * 100
                y_variance = pca_result.explained_variance[y_component] * 100
                pca_figure = px.scatter(
                    pca_plot_data,
                    x=x_component,
                    y=y_component,
                    color=color_column if color_column != "None" else None,
                    hover_name="sample_id",
                    labels={
                        x_component: f"{x_component} ({x_variance:.1f}% explained variance)",
                        y_component: f"{y_component} ({y_variance:.1f}% explained variance)",
                    },
                )
                st.plotly_chart(pca_figure, use_container_width=True)

                st.markdown("### Sample outliers")
                outlier_defaults = OutlierConfig()
                outlier_controls = st.columns(2)
                distance_threshold = outlier_controls[0].slider(
                    "PCA distance percentile threshold",
                    min_value=90.0,
                    max_value=100.0,
                    value=float(outlier_defaults.distance_percentile_threshold),
                    step=0.5,
                )
                contamination = outlier_controls[1].slider(
                    "Isolation Forest contamination",
                    min_value=0.01,
                    max_value=0.50,
                    value=float(outlier_defaults.isolation_contamination),
                    step=0.01,
                )
                outlier_result = analyze_outliers(
                    pca_result.coordinates,
                    OutlierConfig(
                        distance_percentile_threshold=distance_threshold,
                        isolation_contamination=contamination,
                    ),
                )
                outlier_table = outlier_result.results.reset_index()
                st.dataframe(outlier_table, use_container_width=True, hide_index=True)
                for finding in outlier_result.findings:
                    with st.expander(f"[{finding.severity.upper()}] {finding.message}"):
                        render_finding_details(finding)
            except ValueError as exc:
                st.warning(f"PCA could not be calculated: {exc}")

        st.markdown("## Label Consistency")
        if not categorical_columns:
            st.info("No categorical metadata fields are available for label consistency analysis.")
        else:
            label_defaults = LabelConsistencyConfig()
            label_controls = st.columns(3)
            label_column = label_controls[0].selectbox(
                "Metadata field",
                categorical_columns,
                key="label_consistency_column",
            )
            include_classifier = label_controls[1].checkbox(
                "Run cross-validated Logistic Regression",
                value=True,
            )
            minimum_class_samples = label_controls[2].number_input(
                "Minimum samples per class",
                min_value=2,
                max_value=max(dataset.n_samples, 2),
                value=min(label_defaults.min_samples_per_class, max(dataset.n_samples, 2)),
            )
            try:
                label_result = analyze_label_consistency(
                    dataset.expression,
                    dataset.metadata,
                    label_column,
                    LabelConsistencyConfig(min_samples_per_class=int(minimum_class_samples)),
                    include_classifier=include_classifier,
                )
                label_table = label_result.centroid.results.copy()
                label_table["similarity_values"] = label_table["similarity_values"].map(
                    lambda values: ", ".join(
                        f"{group}: {value:.4f}" if value is not None else f"{group}: unavailable"
                        for group, value in values.items()
                    )
                )

                classification = label_result.cross_validated
                if classification is not None and classification.available:
                    label_table = label_table.join(
                        classification.results[["cross_validated_predicted_class", "confidence"]],
                        how="left",
                    )
                else:
                    label_table["cross_validated_predicted_class"] = None
                    label_table["confidence"] = np.nan
                    if classification is not None and classification.reason:
                        st.info(classification.reason)

                display_columns = [
                    "recorded_group",
                    "molecular_closest_group",
                    "cross_validated_predicted_class",
                    "confidence",
                    "recorded_group_similarity",
                    "closest_group_similarity",
                    "similarity_margin",
                    "appears_more_similar_to_another_group",
                    "similarity_values",
                ]
                st.dataframe(
                    label_table[display_columns].reset_index(),
                    use_container_width=True,
                    hide_index=True,
                )

                label_findings = list(label_result.centroid.findings)
                if classification is not None:
                    label_findings.extend(classification.findings)
                if label_findings:
                    st.markdown("### Label consistency evidence")
                    for finding in label_findings:
                        with st.expander(finding.message):
                            render_finding_details(finding)

                if classification is not None and classification.available:
                    st.markdown("### Cross-validated confusion matrix")
                    confusion = pd.crosstab(
                        classification.results["recorded_class"],
                        classification.results["cross_validated_predicted_class"],
                    )
                    confusion_figure = px.imshow(
                        confusion,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels={
                            "x": "Cross-validated predicted group",
                            "y": "Recorded group",
                            "color": "Samples",
                        },
                        aspect="auto",
                    )
                    st.plotly_chart(confusion_figure, use_container_width=True)
            except ValueError as exc:
                st.warning(f"Label consistency could not be calculated: {exc}")

        st.markdown("## Batch Effects")
        if not categorical_columns:
            st.info("No categorical metadata fields are available for batch-effect analysis.")
        else:
            batch_column = st.selectbox(
                "Batch metadata field",
                categorical_columns,
                index=categorical_columns.index("batch") if "batch" in categorical_columns else 0,
            )
            try:
                usable_gene_count = int(dataset.expression.var(axis=1, ddof=0).gt(0).sum())
                batch_component_count = min(3, dataset.n_samples, usable_gene_count)
                if batch_component_count < 2:
                    raise ValueError("batch PCA requires at least two usable genes and samples")
                batch_pca = run_pca(
                    dataset.expression,
                    PCAConfig(remove_zero_variance_genes=True, n_components=batch_component_count),
                )
                batch_result = analyze_batch_pca_association(
                    batch_pca.coordinates,
                    dataset.metadata,
                    batch_column,
                )

                batch_plot_data = batch_pca.coordinates.reset_index()
                batch_metadata = dataset.metadata.drop_duplicates("sample_id", keep="first").copy()
                batch_metadata["sample_id"] = batch_metadata["sample_id"].astype(str)
                batch_plot_data = batch_plot_data.merge(
                    batch_metadata[["sample_id", batch_column]],
                    on="sample_id",
                    how="left",
                )
                batch_x_variance = batch_pca.explained_variance["PC1"] * 100
                batch_y_variance = batch_pca.explained_variance["PC2"] * 100
                batch_pca_figure = px.scatter(
                    batch_plot_data,
                    x="PC1",
                    y="PC2",
                    color=batch_column,
                    hover_name="sample_id",
                    labels={
                        "PC1": f"PC1 ({batch_x_variance:.1f}% explained variance)",
                        "PC2": f"PC2 ({batch_y_variance:.1f}% explained variance)",
                    },
                )
                st.plotly_chart(batch_pca_figure, use_container_width=True)

                batch_summary_columns = st.columns(2)
                batch_summary_columns[0].metric("Batch Effect Risk", batch_result.risk)
                batch_summary_columns[1].metric(
                    "Maximum effect size",
                    f"{batch_result.evidence['maximum_effect_size']:.3f}",
                )
                st.write("Batch sample counts:", batch_result.batch_counts)
                st.dataframe(
                    batch_result.associations.reset_index(),
                    use_container_width=True,
                    hide_index=True,
                )
                for finding in batch_result.findings:
                    with st.expander(f"[{finding.severity.upper()}] {finding.message}"):
                        render_finding_details(finding)
                st.caption("BioDetective reports batch evidence only and does not alter or correct expression values.")
            except ValueError as exc:
                st.warning(f"Batch effects could not be calculated: {exc}")

        st.markdown("## Confounding Analysis")
        if len(categorical_columns) < 2:
            st.info("At least two categorical metadata fields are required for confounding analysis.")
        else:
            confounding_controls = st.columns(2)
            biological_default = categorical_columns.index("condition") if "condition" in categorical_columns else 0
            biological_column = confounding_controls[0].selectbox(
                "Biological variable",
                categorical_columns,
                index=biological_default,
            )
            technical_options = [column for column in categorical_columns if column != biological_column]
            technical_default = technical_options.index("batch") if "batch" in technical_options else 0
            technical_column = confounding_controls[1].selectbox(
                "Technical variable",
                technical_options,
                index=technical_default,
            )
            try:
                confounding_result = analyze_confounding(
                    dataset.metadata,
                    biological_column,
                    technical_column,
                )
                contingency = confounding_result.contingency

                st.markdown("### Contingency table")
                st.dataframe(contingency.contingency_table, use_container_width=True)

                confounding_plots = st.columns(2)
                with confounding_plots[0]:
                    st.markdown("### Count heatmap")
                    contingency_heatmap = px.imshow(
                        contingency.contingency_table,
                        text_auto=True,
                        color_continuous_scale="Blues",
                        labels={
                            "x": technical_column,
                            "y": biological_column,
                            "color": "Samples",
                        },
                        aspect="auto",
                    )
                    st.plotly_chart(contingency_heatmap, use_container_width=True)

                with confounding_plots[1]:
                    st.markdown("### Technical proportions within biological groups")
                    proportions = (
                        contingency.technical_given_biological
                        .reset_index()
                        .melt(
                            id_vars=biological_column,
                            var_name=technical_column,
                            value_name="proportion",
                        )
                    )
                    proportion_figure = px.bar(
                        proportions,
                        x=biological_column,
                        y="proportion",
                        color=technical_column,
                        barmode="stack",
                        labels={"proportion": "Conditional proportion"},
                    )
                    st.plotly_chart(proportion_figure, use_container_width=True)

                confounding_metrics = st.columns(3)
                confounding_metrics[0].metric("Confounding Risk", confounding_result.risk)
                confounding_metrics[1].metric("Cramer's V", f"{confounding_result.cramers_v:.3f}")
                confounding_metrics[2].metric(
                    "Chi-square p-value",
                    f"{confounding_result.p_value:.3g}" if confounding_result.p_value is not None else "Not applicable",
                )
                st.markdown("**Observation**")
                st.write(f"The selected variables have a confounding risk rating of {confounding_result.risk}.")
                st.markdown("**Evidence**")
                st.json(confounding_result.evidence)
                st.markdown("**Interpretation**")
                st.write(confounding_result.interpretation)
                st.markdown("**Recommendation**")
                st.write(
                    "Review the study design and conditional group counts before attributing variation to either variable; "
                    "additional balanced data may be required when separation is limited."
                )
                if confounding_result.sparse_table:
                    st.info("The contingency table is sparse; interpret the chi-square result cautiously.")
            except ValueError as exc:
                st.warning(f"Confounding analysis could not be calculated: {exc}")
    except DataLoadError as exc:
        st.error(str(exc))
