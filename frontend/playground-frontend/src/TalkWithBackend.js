export const train_and_output = (
  user_arch,
  criterion,
  optimizerName,
  problemType,
  targetCol = null,
  features = null,
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
