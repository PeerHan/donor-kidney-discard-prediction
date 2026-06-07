import os
import argparse
import sys

from preprocessing import DataPreprocessor
from merge_excels import DataMerger


def main():
    # Create a train-test-split
    os.system("python src/data/create_train_test_split.py")
    os.system("python src/data/nlp_ecmo.py")

    # Merge all excels
    print("Merging")
    merger = DataMerger()
    merger.perform('src/data/complete_config.yml')

    # Preprocessing and Feature Engineering
    print("Processing Domain Features")
    processor_domain = DataPreprocessor(target='Transplantationsstatus')
    processor_domain.perform('src/data/domain_config.yml',
                      test_split=True,
                      dev=False,
                      impute=True)
    print("Processing Exploration Features")
    processor_exploration = DataPreprocessor(target='Transplantationsstatus')
    processor_exploration.perform('src/data/exploration_config.yml',
                      test_split=True,
                      dev=False,
                      impute=True)


if __name__ == "__main__":
    main()
