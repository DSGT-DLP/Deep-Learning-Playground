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
  batchSize,
  shuffle,
  csvData = null,
  fileURL = null,
  email
) => {
  const runResult = await fetch("/run", {
    method: "POST",
    body: JSON.stringify({
      user_arch: user_arch,
      criterion: criterion,
      optimizer_name: optimizerName,
      problem_type: problemType,
      target: targetCol,
      features: features,
      default: usingDefaultDataset,
      test_size: testSize,
      epochs: epochs,
      batch_size: batchSize,
      shuffle: shuffle,
      csvData: csvData,
      fileURL: fileURL,
      email: email,
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
    },
  })
    .then((res) => res.json())
    .then((data) => data)
    .catch((error) => error);

  if (runResult.success) {
    // send email if provided
    if (email?.length) {
      await fetch("/sendemail", {
        method: "POST",
        body: JSON.stringify({
          email_address: email,
          subject:
            "Your output files and visualizations from Deep Learning Playground",
          body_text:
            "Attached are the output files and visualizations that you just created in Deep Learning Playground on datasciencegt-dlp.com. Please notify us if there are any problems.",
          attachment_array: [
              // we will not create constant values for the source files because the constants cannot be used in Home
              "../frontend/playground-frontend/src/backend_outputs/my_deep_learning_model.onnx",
              "../frontend/playground-frontend/src/backend_outputs/model.pt",
            "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_loss_plot.png",
            "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_accuracy_plot.png",
            "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_confusion_matrix.png",
            "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_AUC_ROC_Curve.png",
          ],
          }),
        
      });
    }
  }

  return runResult;
};
