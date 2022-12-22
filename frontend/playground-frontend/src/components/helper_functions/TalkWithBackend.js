import { toast } from "react-toastify";
import { auth } from "../../firebase";
import axios from "axios";

const uploadToBackend = async (data) => {
  let headers = auth.currentUser
    ? { Authorization: "bearer " + (await auth.currentUser.getIdToken(true)) }
    : undefined;

  await axios.post("/api/upload", data, { headers });
};

const userCodeEval = async (file, snippet) => {
  const codeEval = await sendToBackend("sendUserCodeEval", {
    codeSnippet: snippet,
    file: file,
  });
  return codeEval;
};

const getSignedUploadUrl = async (version, filename) => {
  return await sendToBackend("getSignedUploadUrl", {
    filename,
    version,
  });
};

const sendToBackend = async (route, data) => {
  let headers = auth.currentUser
    ? { Authorization: "bearer " + (await auth.currentUser.getIdToken(true)) }
    : undefined;
  const backendResult = await fetch(`/api/${route}`, {
    method: "POST",
    body: JSON.stringify(data),
    headers: headers,
  }).then((result) => result.json());
  return backendResult;
};

const routeDict = {
  tabular: "tabular-run",
  image: "img-run",
  pretrained: "pretrain-run",
};

const train_and_output = async (choice, choiceDict) => {
  const trainResult = await sendToBackend(routeDict[choice], choiceDict);
  return trainResult;
};

const sendEmail = async (email, problemType) => {
  // send email if provided
  const attachments = [
    // we will not create constant values for the source files because the constants cannot be used in Home
    "./frontend/playground-frontend/src/backend_outputs/my_deep_learning_model.onnx",
    "./frontend/playground-frontend/src/backend_outputs/model.pt",
    "./frontend/playground-frontend/src/backend_outputs/visualization_output/my_loss_plot.png",
  ];

  if (problemType === "classification") {
    attachments.push(
      "./frontend/playground-frontend/src/backend_outputs/visualization_output/my_accuracy_plot.png"
    );
    attachments.push(
      "./frontend/playground-frontend/src/backend_outputs/visualization_output/my_confusion_matrix.png"
    );
    attachments.push(
      "./frontend/playground-frontend/src/backend_outputs/visualization_output/my_AUC_ROC_Curve.png"
    );
  }

  const emailResult = await sendToBackend("sendEmail", {
    email_address: email,
    subject:
      "Your output files and visualizations from Deep Learning Playground",
    body_text:
      "Attached are the output files and visualizations that you just created in Deep Learning Playground on datasciencegt-dlp.com. Please notify us if there are any problems.",
    attachment_array: attachments,
  });

  if (!emailResult.success) {
    toast.error(emailResult.message);
  }
};

const isLoggedIn = async () => {
  return await auth.currentUser?.getIdToken(true), toast.error("Not logged in");
};

export {
  uploadToBackend,
  sendToBackend,
  train_and_output,
  sendEmail,
  isLoggedIn,
  userCodeEval,
  getSignedUploadUrl,
};
