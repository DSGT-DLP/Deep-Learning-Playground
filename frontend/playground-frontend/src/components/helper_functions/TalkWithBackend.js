import { io } from "socket.io-client";
import { toast } from "react-toastify";

<<<<<<< HEAD
const socketEventDict = {
  tabular: "runTraining",
  image: "img-run",
  pretrained: "pretrain-run",
};

const socket = io(":5000", { timeout: 60000 });
=======
const socket = io(":8000");
>>>>>>> c7a2a4dd0c415944be20e103049aa8d4c48d8dd0
socket.on("connect", () => {
  frontendLog(`connected to socket`);
});
socket.on("connect_error", () => {
  socket.close();
});

const frontendLog = (log) => {
  socket.emit("frontendLog", log);
};

const train_and_output = (choice, choiceDict) => {
  socket.emit(socketEventDict[choice], choiceDict);
};

const sendEmail = (email, problemType) => {
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

  socket.emit("sendEmail", {
    email_address: email,
    subject:
      "Your output files and visualizations from Deep Learning Playground",
    body_text:
      "Attached are the output files and visualizations that you just created in Deep Learning Playground on datasciencegt-dlp.com. Please notify us if there are any problems.",
    attachment_array: attachments,
  });
};

socket.on("emailResult", (result) => {
  if (!result.success) {
    toast.error(result.message);
  }
});

export { socket, frontendLog, train_and_output, sendEmail };
