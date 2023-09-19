# Examples of using the DLP backend

Go to [`app.py`](./app.py) to implement these examples.

## Deep Learning

```
print(
    dl_drive(
        ["nn.Linear(8, 10)", "nn.ReLU()", "nn.Linear(10, 1)"],
        "MSELOSS",
        "SGD",
        problem_type="regression",
        default=True,
        epochs=10,
    )
```

```
print(
    dl_drive(
        ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"],
        "CELOSS",
        "SGD",
        problem_type="classification",
        default=False,
        epochs=10,
    )
)
```

## Machine Learning

```
print(ml_drive("DecisionTreeClassifier(max_depth=3, random_state=15)",
        problem_type="classification", default=True))
```
