from mas_deepr.data.train_pool import assign_split


def test_assign_split_deterministic() -> None:
    assert assign_split("musique-123", val_fraction=0.2) == assign_split(
        "musique-123", val_fraction=0.2
    )


def test_assign_split_distribution_close_to_fraction() -> None:
    val_fraction = 0.2
    n = 5000
    val_count = sum(
        1
        for i in range(n)
        if assign_split(f"q-{i}", val_fraction=val_fraction) == "val"
    )
    ratio = val_count / n
    assert abs(ratio - val_fraction) < 0.03


def test_assign_split_only_two_labels() -> None:
    labels = {assign_split(f"q-{i}", val_fraction=0.3) for i in range(200)}
    assert labels <= {"train", "val"}
