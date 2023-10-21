from training.core.dataset import SklearnDatasetCreator
from torch.utils.data import DataLoader


def test_iris_train_test_split():
    dataset_name = "IRIS"
    test_size = 0.2
    shuffle = True

    dataCreator = SklearnDatasetCreator.fromDefault(dataset_name, test_size, shuffle)

    X_train, X_test, y_train, y_test = (
        dataCreator._X_train,
        dataCreator._X_test,
        dataCreator._y_train,
        dataCreator._y_test,
    )

    total_dataset_size = len(dataCreator.getDefaultDataset(dataset_name))

    expected_size = int(total_dataset_size * test_size)

    assert len(X_test) == expected_size
    assert len(y_test) == expected_size


def test_iris_train_test_loader():
    dataset_name = "IRIS"
    test_size = 0.2
    shuffle = True
    batch_size = 20
    drop_last = True

    dataCreator = SklearnDatasetCreator.fromDefault(dataset_name, test_size, shuffle)

    train_loader = DataLoader(
        dataCreator.createTrainDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    test_loader = DataLoader(
        dataCreator.createTestDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    # Check that all batches in the train loader and test loader have size batch_size and there are the right number of batches
    X_train, X_test, y_train, y_test = (
        dataCreator._X_train,
        dataCreator._X_test,
        dataCreator._y_train,
        dataCreator._y_test,
    )

    total_dataset_size = len(dataCreator.getDefaultDataset(dataset_name))

    test_len = total_dataset_size * test_size
    train_len = total_dataset_size - test_len

    num_train_batches_expected = train_len // batch_size
    num_test_batches_expected = test_len // batch_size

    train_batch_count = 0
    test_batch_count = 0

    for batch in train_loader:
        train_features, train_labels = batch
        train_batch_count += 1
        assert train_features.size()[0] == batch_size

    assert train_batch_count == num_train_batches_expected

    for batch in test_loader:
        test_features, test_labels = batch
        test_batch_count += 1
        assert test_features.size()[0] == batch_size

    assert test_batch_count == num_test_batches_expected
