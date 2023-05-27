# Unit tests for Dynamo DB Files
The following are some unit tests for the current dynamo Db util files for each of these tables:

## dynamo_db_utils.py
```py
if __name__ == "__main__":
    print(1)
    print(2, get_dynamo_item_by_id("trainspace", "blah"))
    print(3, get_dynamo_items_by_gsi("trainspace", "bleh"))
    print(
        4,
        create_dynamo_item(
            "trainspace",
            {
                "trainspace_id": str(random.random()),
                "uid": "bleh",
                "created": datetime.now().isoformat(),
            },
        ),
    )
    print(
        5,
        update_dynamo_item(
            "trainspace",
            "0.6637985062827166",
            {"uid": "blah", "created": datetime.now().isoformat()},
        ),
    )
    print(6, delete_dynamo_item("trainspace", "ergsdf"))
```

## execution_db.py
```py
if __name__ == "__main__":
    print(1)
    print(2, getAllUserExecutionData("8hDeAbdZ9Lg301QFGdEYYeAq4Kw2"))
    print(3, getExecutionData("exfddc9ad2666d31cae1790167aefc9aa34eb5d06a28e1805e8fa8881845d463a8"))
    print(3, updateExecutionData("exfddc9ad2666d31cae1790167aefc9aa34eb5d06a28e1805e8fa8881845d463a8", {
        "timestamp": datetime.now().isoformat(),
    }))
    print(4, createExecutionData(
        ExecutionData(
            execution_id=str(random.random()),
            data_source='TABULAR',
            name='hola',
            status='QUEUED',
            timestamp=str(datetime.now().isoformat()),
            user_id='bleh'
    )))
```

## file_upload_db.py
```py
if __name__ == "__main__":
    print(0)
    print(1, getFileUploadData("test011"))
    print(2, updateFileUploadData("test011", {"s3_uri": "test02"}))
    print(3, createFileUploadData(FileUploadData("test012", "bleh")))
    print(5, getAllUserFileUploadData("bleh"))
```

## trainspace.py
```py
if __name__ == "__main__":
    print(1)
    print(2, getTrainspaceData("e4d46926-1eaa-42b0-accb-41a3912038e4"))
    print(
        3,
        updateTrainspaceData(
            "e4d46926-1eaa-42b0-accb-41a3912038e4",
            {"created": datetime.now().isoformat()},
        ),
    )
    print(4, getAllUserTrainspaceData("bleh"))
    print(
        5,
        createTrainspaceData(
            TrainspaceData(
                trainspace_id=str(random.random()),
                created=datetime.now().isoformat(),
                data_source="TABULAR",
                dataset_data={
                    "name": {"S": "IRIS"},
                    "is_default_dataset": {"BOOL": True},
                },
                name=str(random.random()),
                parameters_data={
                    "features": {
                        "L": [
                            {"S": "sepal length (cm)"},
                            {"S": "sepal width (cm)"},
                            {"S": "petal length (cm)"},
                            {"S": "petal width (cm)"},
                        ]
                    },
                    "criterion": {"S": "CELOSS"},
                    "batch_size": {"N": "20"},
                    "test_size": {"N": "0.2"},
                    "target_col": {"S": "target"},
                    "layers": {
                        "L": [
                            {
                                "M": {
                                    "value": {"S": "LINEAR"},
                                    "parameters": {"L": [{"N": "10"}, {"N": "3"}]},
                                }
                            },
                            {"M": {"value": {"S": "RELU"}, "parameters": {"L": []}}},
                            {
                                "M": {
                                    "value": {"S": "LINEAR"},
                                    "parameters": {"L": [{"N": "3"}, {"N": "10"}]},
                                }
                            },
                            {
                                "M": {
                                    "value": {"S": "SOFTMAX"},
                                    "parameters": {"L": [{"N": "-1"}]},
                                }
                            },
                        ]
                    },
                    "problem_type": {"S": "CLASSIFICATION"},
                    "shuffle": {"BOOL": True},
                    "epochs": {"N": "5"},
                    "optimizer_name": {"S": "SGD"},
                },
                review_data={
                    "notification_phone_number": {"NULL": True},
                    "notification_email": {"NULL": True},
                },
                status="QUEUED",
                uid="bleh",
            )
        ),
    )
```

## userprogress_db.py
```py
if __name__ == "__main__":
    print(1)
    print(2, getAllUserProgressData("LTLZSmoEnYQc9Kx7xJ3Zygwojro2"))
    print(3, createUserProgressData(UserProgressData("bleh", {"hola": 1})))
    print(4, updateUserProgressData("bleh", {"progressData": {"hola": 2}}))
```