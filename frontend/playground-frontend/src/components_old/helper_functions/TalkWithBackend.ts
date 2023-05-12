import axios from "axios";
import sha256 from "crypto-js/sha256";
import { toast } from "react-toastify";
import { EXPECTED_FAILURE_HTTP_CODES, ROUTE_DICT } from "../../constants";
import { auth } from "../../common/utils/firebase";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function uploadToBackend(data: { [key: string]: any }) {
  const headers = auth.currentUser
    ? { Authorization: "Bearer " + (await auth.currentUser.getIdToken(true)) }
    : undefined;
  await axios.post("/api/upload", data, { headers });
}

export const userCodeEval = async (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: { [key: string]: any },
  snippet: string
) => {
  const codeEval = await sendToBackend("sendUserCodeEval", {
    data: data,
    codeSnippet: snippet,
  });
  return codeEval;
};

export const getSignedUploadUrl = async (
  version: number,
  filename: string,
  file: File
) => {
  const headers = auth.currentUser
    ? { Authorization: "Bearer " + (await auth.currentUser.getIdToken(true)) }
    : undefined;
  const data = new FormData();
  data.append("version", version.toString());
  data.append("filename", filename);
  data.append("file", file);
  return await fetch("/api/getSignedUploadUrl", {
    method: "POST",
    body: data,
    headers: headers,
  });
};

/**
 * Given timestamp and unique user id, generate an execution id
 * @param uid
 * @param timestamp
 * @returns execution id
 */
function createExecutionId(
  uid: string,
  timestamp: string = new Date().getTime().toString()
) {
  const hash = sha256(timestamp + uid);
  return "ex" + hash;
}

export interface JSONResponseType {
  success: boolean;
  message: string;
}
export async function sendToBackend(
  route: string,
  data: { [key: string]: unknown }
) {
  if (auth.currentUser == null) throw new Error("Not logged in");

  const headers = {
    Authorization: "Bearer " + (await auth.currentUser.getIdToken(true)),
    uid: auth.currentUser.uid,
  };
  data["route"] = route;

  data["execution_id"] = createExecutionId(headers.uid);
  data["user"] = {
    uid: auth.currentUser.uid,
    email: auth.currentUser.email,
    displayName: auth.currentUser.displayName,
  };
  if (data.shouldBeQueued) {
    const backendResult = await fetch("/api/writeToQueue", {
      method: "POST",
      body: JSON.stringify(data),
      headers: headers,
    }).then((result) => result.json());
    return backendResult;
  }
  const backendResult = await fetch(`/api/${route}`, {
    method: "POST",
    body: JSON.stringify(data),
    headers: headers,
  }).then((result) => {
    if (result.ok) return result.json();
    else if (EXPECTED_FAILURE_HTTP_CODES.includes(result.status)) {
      return result.json().then((json: JSONResponseType) => {
        toast.error(json.message);
        throw new Error(json.message);
      });
    } else if (result.status === 504) {
      toast.error("Backend not active. Please try again later.");
      throw new Error("Something went wrong. Please try again later.");
    } else {
      toast.error("Something went wrong. Please try again later.");
      throw new Error("Something went wrong. Please try again later.");
    }
  });
  return backendResult;
}

export async function train_and_output(
  choice: keyof typeof ROUTE_DICT,
  data: { [key: string]: unknown }
) {
  const route = ROUTE_DICT[choice];

  if (process.env.REACT_APP_MODE === "prod") {
    data["shouldBeQueued"] = true;
  }
  const trainResult = await sendToBackend(route, data);
  return trainResult as JSONResponseType;
}

export async function sendEmail(
  email: string,
  problemType: "classification" | "regression"
) {
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

export async function isLoggedIn() {
  return await auth.currentUser?.getIdToken(true), toast.error("Not logged in");
}
