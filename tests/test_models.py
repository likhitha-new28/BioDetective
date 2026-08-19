import pandas as pd

from biodetective.core.models import BioDataset


def test_biodataset_helper_properties():
    expression = pd.DataFrame({"S01": [10, 5], "S02": [12, 6]}, index=["TP53", "BRCA1"])
    metadata = pd.DataFrame({"sample_id": ["S01", "S02"], "condition": ["Healthy", "Cancer"]})

    dataset = BioDataset(expression, metadata, name="demo")

    assert dataset.sample_ids == ["S01", "S02"]
    assert dataset.feature_ids == ["TP53", "BRCA1"]
    assert dataset.n_samples == 2
    assert dataset.n_features == 2
    assert dataset.metadata_columns == ["sample_id", "condition"]
    assert dataset.name == "demo"
