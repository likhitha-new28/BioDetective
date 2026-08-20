"""Streamlit page for explicit, metadata-only GEO dataset discovery."""

import pandas as pd
import streamlit as st

from biodetective.integrations.geo import GEOIntegrationError, fetch_geo_metadata
from biodetective.ui import apply_responsive_styles


st.set_page_config(page_title="BioDetective — GEO Import", page_icon="🧬", layout="wide")
apply_responsive_styles()
st.title("GEO Import")
st.caption("Discover public GEO metadata and files. BioDetective will not download or analyze a dataset automatically.")

accession = st.text_input("GEO Series accession", placeholder="GSE1000").strip().upper()
if st.button("Fetch GEO metadata", type="primary"):
    try:
        with st.spinner("Fetching metadata from NCBI GEO..."):
            st.session_state["geo_metadata"] = fetch_geo_metadata(accession)
            st.session_state["geo_accession"] = accession
    except GEOIntegrationError as exc:
        st.session_state.pop("geo_metadata", None)
        st.session_state.pop("geo_accession", None)
        st.error(str(exc))

record = st.session_state.get("geo_metadata")
if record is not None and st.session_state.get("geo_accession") == accession:
    st.success(f"Loaded metadata for {record.accession}.")
    st.markdown(f"## {record.title}")
    st.write(record.description or "No description was provided by GEO.")

    summary = st.columns(4)
    summary[0].metric("Accession", record.accession)
    summary[1].metric("Platforms", len(record.platforms))
    summary[2].metric("Samples", len(record.samples))
    summary[3].metric("Discoverable files", len(record.expression_files) + len(record.supplementary_files))

    st.markdown("## Available platforms")
    if record.platforms:
        st.dataframe(pd.DataFrame({"platform_accession": record.platforms}), use_container_width=True, hide_index=True)
    else:
        st.info("No platform accessions were listed in the GEO Series metadata.")

    selected_platform = None
    if len(record.platforms) > 1:
        platform_choice = st.selectbox(
            "Select a platform",
            ["Select a platform...", *record.platforms],
            help="This Series contains multiple platforms; choose one explicitly before selecting samples or files.",
        )
        if platform_choice != "Select a platform...":
            selected_platform = platform_choice
        else:
            st.warning("Select a platform to resolve the multi-platform mapping.")
    elif len(record.platforms) == 1:
        selected_platform = record.platforms[0]
        st.write(f"Selected platform: **{selected_platform}**")

    st.markdown("## Available samples")
    sample_records = [
        {
            "sample_accession": sample.accession,
            "title": sample.title,
            "platform": sample.platform_accession,
            "supplementary_files": len(sample.supplementary_files),
        }
        for sample in record.samples
    ]
    st.dataframe(pd.DataFrame(sample_records), use_container_width=True, hide_index=True)

    platform_is_resolved = len(record.platforms) <= 1 or selected_platform is not None
    selectable_samples = [
        sample
        for sample in record.samples
        if selected_platform is None or sample.platform_accession == selected_platform
    ]
    if platform_is_resolved and len(selectable_samples) > 1:
        selected_samples = st.multiselect(
            "Select samples",
            [sample.accession for sample in selectable_samples],
            default=[],
            help="Multiple samples are available. Choose the samples you intend to use; none are selected automatically.",
        )
        if not selected_samples:
            st.info("Select one or more samples to continue with this GEO source.")
    elif platform_is_resolved and len(selectable_samples) == 1:
        st.write(f"Selected sample: **{selectable_samples[0].accession}**")

    st.markdown("## Expression and supplementary files")
    expression_records = [
        {
            "name": item.name,
            "platform": item.platform_accession or "Ambiguous/not specified",
            "type": item.kind,
            "url": item.url,
        }
        for item in record.expression_files
    ]
    supplementary_items = [*record.supplementary_files]
    for sample in record.samples:
        supplementary_items.extend(sample.supplementary_files)
    supplementary_records = [
        {
            "name": item.name,
            "source": item.source_accession,
            "platform": item.platform_accession or "Not specified",
            "type": item.kind,
            "url": item.url,
        }
        for item in supplementary_items
    ]
    if expression_records:
        st.markdown("### SeriesMatrix expression files")
        st.dataframe(pd.DataFrame(expression_records), use_container_width=True, hide_index=True)
        eligible_expression_files = [
            item
            for item in record.expression_files
            if selected_platform is None or item.platform_accession in {None, selected_platform}
        ]
        if platform_is_resolved and len(eligible_expression_files) > 1:
            expression_choice = st.selectbox(
                "Select an expression file",
                ["Select an expression file...", *[item.name for item in eligible_expression_files]],
            )
            if expression_choice == "Select an expression file...":
                st.info("Choose the intended expression file; BioDetective will not guess between multiple files.")
        elif platform_is_resolved and len(eligible_expression_files) == 1:
            st.write(f"Selected expression file: **{eligible_expression_files[0].name}**")
    else:
        st.info("No SeriesMatrix expression files were discoverable for this accession.")

    if supplementary_records:
        st.markdown("### Supplementary files")
        st.dataframe(pd.DataFrame(supplementary_records), use_container_width=True, hide_index=True)
        if len(supplementary_items) > 1:
            selected_supplementary = st.multiselect(
                "Select supplementary files",
                [f"{item.source_accession}: {item.name}" for item in supplementary_items],
                default=[],
            )
            if not selected_supplementary:
                st.info("Multiple supplementary files are available; select files explicitly if needed.")
        else:
            st.write(f"Selected supplementary file: **{supplementary_items[0].name}**")

    st.warning("GEO discovery does not infer gene identifiers, sample-to-condition mappings, or the correct file when alternatives exist.")
