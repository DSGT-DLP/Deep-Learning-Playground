const routeDict = {
  "tabular": "/run",
  "image": "/img-run",
  "pretrained": "/pretrain-run"
};

export const train_and_output = async (choice, choiceDict) => {

  const runResult = await fetch(routeDict[choice], {
    method: "POST",
    body: JSON.stringify(choiceDict),
    headers: {
      "Content-type": "application/json; charset=UTF-8",
    },
  })
    .then((res) => res.json())
    .then((data) => data)
    .catch((error) => error);

  if (runResult.success) {
    const email = choiceDict["email"];
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
