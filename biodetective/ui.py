"""Shared Streamlit presentation helpers."""

import streamlit as st


def apply_responsive_styles() -> None:
    """Keep the dashboard usable on narrow mobile browser viewports."""
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stFileUploader"] section {
            min-height: 7rem;
        }
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button,
        .stButton button {
            min-height: 2.75rem;
        }
        @media (max-width: 640px) {
            .block-container {
                padding: 1rem 0.85rem 2rem;
            }
            h1 { font-size: 2rem !important; }
            h2 { font-size: 1.55rem !important; }
            h3 { font-size: 1.25rem !important; }
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.5rem;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
            [data-testid="stDownloadButton"] button,
            [data-testid="stFileUploader"] button,
            .stButton button {
                width: 100%;
            }
            [data-testid="stMetric"] {
                padding: 0.75rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
