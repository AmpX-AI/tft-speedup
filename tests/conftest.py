import pytest

from tf_utils import ColumnTypeInfo, ColumnTypes


def sample_dataset_feature():
    return [
        "target",
        "feature",
        "feature2",
        "feature3",
        "feature_cs",
        "feature2_cs",
        "feature3_cs",
        "feature4",
    ]


"""
@pytest.fixture(scope="session", autouse=False)
def feature_list():
    return sample_dataset_feature()
"""


@pytest.fixture(scope="session", autouse=False)
def input_types():
    columns_ordering = sample_dataset_feature()

    def get_cti(name_list):
        return ColumnTypeInfo(names=name_list, loc=[columns_ordering.index(name) for name in name_list])

    return ColumnTypes(
        known_inputs=get_cti(["feature_cs", "feature2_cs", "feature3_cs"]),
        observed_inputs=get_cti(["feature4"]),
        forecast_inputs=get_cti(["feature", "feature2", "feature3"]),
        static_inputs=ColumnTypeInfo(),
    )


@pytest.fixture(scope="session", autouse=False)
def input_types_statics():
    columns_ordering = sample_dataset_feature()

    def get_cti(name_list):
        return ColumnTypeInfo(names=name_list, loc=[columns_ordering.index(name) for name in name_list])

    return ColumnTypes(
        known_inputs=get_cti(["feature_cs", "feature2_cs", "feature3_cs"]),
        observed_inputs=get_cti(["feature4"]),
        forecast_inputs=get_cti(["feature3"]),
        static_inputs=get_cti(["feature", "feature2"]),
    )
