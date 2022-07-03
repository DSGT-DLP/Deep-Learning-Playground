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
        }),
      });
    }
  }

  return runResult;
};
