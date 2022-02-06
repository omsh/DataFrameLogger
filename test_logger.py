from logger import DFLogger


def test_parse_column_names():
    df_logger = DFLogger(column_names="epoch | accuracy", sep="|")
    assert len(df_logger.column_names) != 0
