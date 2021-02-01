import pandas


def number_classes(column):
    label_mapping = {label: i for i, label in enumerate(column.unique())}
    return label_mapping
