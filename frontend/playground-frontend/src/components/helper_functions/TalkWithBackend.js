import { io } from "socket.io-client";
import { toast } from "react-toastify";
import { auth } from "../../firebase";

const socketEventDict = {
  tabular: "tabular-run",
  image: "img-run",
  audio: "audio-run",
  pretrained: "pretrain-run",
};

const socket = io();
socket.on("connect", () => {
  frontendLog(`connected to socket`);
});
socket.on("connect_error", (err) => {
  console.log(`connection error due to: ${err.message}`);
  socket.close();
});

const frontendLog = (log) => {
  socket.emit("frontendLog", log);
};

const train_and_output = (choice, choiceDict) => {
  console.log(choice, choiceDict);
  socket.emit(socketEventDict[choice], choiceDict, socket.id);
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

  socket.emit(
    "sendEmail",
    {
      email_address: email,
      subject:
        "Your output files and visualizations from Deep Learning Playground",
      body_text:
        "Attached are the output files and visualizations that you just created in Deep Learning Playground on datasciencegt-dlp.com. Please notify us if there are any problems.",
      attachment_array: attachments,
    },
    socket.id
  );
};

const updateUserSettings = async () => {
  if (auth.currentUser) {
    socket.emit("updateUserSettings", {
      authorization: await auth.currentUser.getIdToken(true),
    });
  } else {
    toast.error("Not logged in");
  }
};

socket.on("emailResult", (result) => {
  if (!result.success) {
    toast.error(result.message);
  }
});

export { socket, frontendLog, train_and_output, sendEmail, updateUserSettings };
