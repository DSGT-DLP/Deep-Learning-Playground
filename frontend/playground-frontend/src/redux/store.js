import { configureStore } from "@reduxjs/toolkit";
import currentUserReducer from "./userLogin";

export default configureStore({
  reducer: {
    currentUser: currentUserReducer,
  },
});
