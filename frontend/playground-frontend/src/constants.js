export const COLORS = {
  input: "#E2E2E2",
  layer: "#CD7BFF",
  addLayer: "#F2ECFF",
  background: "#F6F6FF",
  dark_blue: "#003057",
};

export const LAYOUT = {
  row: {
    display: "flex",
    flexDirection: "row",
  },
  column: {
    display: "flex",
    flexDirection: "column",
  },
  centerMiddle: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};

export const GENERAL_STYLES = {
  p: {
    fontFamily: "Arial, Helvetica, sans-serif",
    fontWeight: "bold",
  },
};

export const ITEM_TYPES = {
  NEW_LAYER: "new_layer",
};

// Vis output paths must be the same as in backend/constants.py
export const LOSS_VIZ =
    "../frontend/playground-frontend/src/visualization_output/my_loss_plot.png",
  ACC_VIZ =
    "../frontend/playground-frontend/src/visualization_output/my_accuracy_plot.png",
  CLASSICAL_ML_CONFUSION_MATRIX =
    "../frontend/playground-frontend/src/visualization_output/confusion_matrix.png";
