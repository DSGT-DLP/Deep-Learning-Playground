export const train_and_output = (
  user_arch,
  criterion,
  optimizer_name,
  problem_type,
  usingDefault,
  epochs
) => {
  fetch("/run", {
    method: "POST",
    body: JSON.stringify({
      user_arch,
      criterion,
      optimizer_name,
      problem_type,
      default: usingDefault,
      epochs,
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
    .then((res) => res.json())
    .then((data) => {
      console.log(data);
    });
};
