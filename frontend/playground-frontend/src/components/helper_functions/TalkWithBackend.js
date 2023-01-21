import { toast } from "react-toastify";
import { auth } from "../../firebase";
import axios from "axios";

async function uploadToBackend(data) {
  let headers = auth.currentUser
    ? { Authorization: "bearer " + (await auth.currentUser.getIdToken(true)) }
    : undefined;

  await axios.post("/api/upload", data, { headers });
}

const userCodeEval = async (data, snippet) => {
  const codeEval = await sendToBackend("sendUserCodeEval", {
    data: data,
    codeSnippet: snippet,
  });
  return codeEval;
};

const getSignedUploadUrl = async (version, filename, file) => {
  let headers = auth.currentUser
    ? { Authorization: "bearer " + (await auth.currentUser.getIdToken(true)) }
    : undefined;
  let data = new FormData();
  data.append("version", version);
  data.append("filename", filename);
  data.append("file", file);
  return await fetch("/api/getSignedUploadUrl", {
    method: "POST",
    body: data,
    headers: headers,
  });
};

async function sendToBackend(route, data) {
  let headers = auth.currentUser
    ? {
        Authorization: "bearer " + (await auth.currentUser.getIdToken(true)),
        uid: auth.currentUser.uid,
      }
    : undefined;
  const backendResult = await fetch(`/api/${route}`, {
    method: "POST",
    body: JSON.stringify(data),
    headers: headers,
  }).then((result) => result.json());
  return backendResult;
}

const routeDict = {
  tabular: "tabular-run",
  image: "img-run",
  pretrained: "pretrain-run",
  classicalml: "ml-run",
  objectdetection: "object-detection",
};

async function train_and_output(choice, choiceDict) {
  if (process.env.MODE === "dev") {
    const trainResult = await sendToBackend(routeDict[choice], choiceDict);
    return trainResult;
  } else {
    //TODO: submit request to sqs. return success or fail message!
  }
}

async function sendEmail(email, problemType) {
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
}

async function isLoggedIn() {
  return await auth.currentUser?.getIdToken(true), toast.error("Not logged in");
}

export {
  uploadToBackend,
  sendToBackend,
  train_and_output,
  sendEmail,
  isLoggedIn,
  userCodeEval,
  getSignedUploadUrl,
};
