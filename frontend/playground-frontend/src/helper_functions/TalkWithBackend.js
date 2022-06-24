export const train_and_output = async (
  user_arch,
  criterion,
  optimizerName,
  problemType,
  targetCol = null,
  features = null,
  usingDefaultDataset = null,
  testSize,
  epochs,
  shuffle,
  csvData = null,
  fileURL = null,
  email
) => {
  const runResult = await fetch("/run", {
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
      csvData,
      fileURL,
      email,
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
    },
  })
    .then((res) => {
      return res.json();
    })
    .then((data) => {
      return data;
    })
    .catch((error) => error);
  if (runResult.success) {
    // send email if provided
    if (email && email.length) {
      await fetch("/sendemail", {
        method: "POST",
        body: JSON.stringify({
          email_address: email,
          subject:
            "Your ONNX file and visualizations from Deep Learning Playground",
          body_text:
            "Attached is the ONNX file and visualizations that you just created in Deep Learning Playground. Please notify us if there are any problems.",
          attachment_array: [
            "../frontend/playground-frontend/src/backend_outputs/my_deep_learning_model.onnx",
            "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_loss_plot.png",
            "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_accuracy_plot.png",
          ], // from backend constants.py
        }),
      });
    }
  }

  return runResult;
};
