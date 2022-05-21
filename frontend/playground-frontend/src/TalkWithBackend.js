export const train_and_output = (
  user_arch,
  criterion,
  optimizerName,
  problemType,
  targetCol,
  features,
  usingDefaultDataset,
  testSize,
  epochs,
  shuffle
) => {
  fetch("/run", {
    method: "POST",
    body: JSON.stringify({
      user_arch,
      criterion,
      optimizer_name: optimizerName,
      problem_type: problemType,
      target: targetCol,
      features,
      default: usingDefaultDataset,
      test_size: testSize,
      epochs,
      shuffle,
      // user_arch: [
      //   "nn.Linear(4, 10)",
      //   "nn.ReLU()",
      //   "nn.Linear(10, 3)",
      //   "nn.Softmax()",
      // ],
      // criterion: "CELOSS",
      // optimizer_name: "SGD",
      // problem_type: "classification",
      // default: true,
      // epochs: 10,
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
    },
  })
    .then((res) => {
      if (res.ok) {
        return res.json();
      }
      throw new Error("Something went wrong");
    })
    .then((data) => console.log(data))
    .catch((error) => console.log(error));
};
