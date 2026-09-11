from miracle_ad.split_registry import (
    Language,
    available_datasets,
    build_partitions,
    load_split,
    relative_audio_path,
)


def test_all_bundled_splits_are_valid_and_disjoint():
    datasets = available_datasets()
    assert {
        "elu",
        "gds3",
        "gpilot",
        "ivanova",
        "mchou",
        "ncmmsc",
        "pitt",
        "taukadial",
        "taukadial_c",
        "taukadial_e",
        "vas",
    } == set(datasets)
    total_train = 0
    total_validation = 0
    for dataset in datasets:
        split = load_split(dataset)
        assert list(split) == ["train", "val", "test"]
        paths = {name: set(entries) for name, entries in split.items()}
        assert paths["train"].isdisjoint(paths["val"])
        assert paths["train"].isdisjoint(paths["test"])
        assert paths["val"].isdisjoint(paths["test"])
        for entries in split.values():
            assert set(entries.values()) <= {0, 1, 2}
        total_train += len(split["train"])
        total_validation += len(split["val"])

    # The legacy training pools contained exactly 2,000 entries. The committed
    # split migration moves 10% of them into validation.
    assert total_train == 1800
    assert total_validation == 200


def test_legacy_paths_are_resolved_portably():
    path = relative_audio_path("../Datasets/Chinese/NCMMSC\\AD\\sample.wav")
    assert path.as_posix() == "Chinese/NCMMSC/AD/sample.wav"


def test_committed_partitions_are_loaded_and_test_is_held_out(tmp_path):
    first = build_partitions("ncmmsc", tmp_path)
    second = build_partitions("ncmmsc", tmp_path)
    for name in first:
        assert [record.source_path for record in first[name]] == [
            record.source_path for record in second[name]
        ]

    paths = {name: {record.source_path for record in records} for name, records in first.items()}
    assert paths["train"].isdisjoint(paths["val"])
    assert paths["train"].isdisjoint(paths["test"])
    assert paths["val"].isdisjoint(paths["test"])


def test_language_ids_are_contiguous():
    assert [language.value for language in Language] == [0, 1, 2, 3]


def test_combined_taukadial_artifact_uses_per_participant_languages(tmp_path):
    partitions = build_partitions("taukadial", tmp_path)
    languages = {
        record.language for records in partitions.values() for record in records
    }
    assert languages == {Language.ENGLISH, Language.CHINESE}
