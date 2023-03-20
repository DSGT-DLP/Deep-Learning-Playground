import { configureStore } from "@reduxjs/toolkit";
import currentUserReducer from "./userLogin";
import trainReducer from "./train";
import { useDispatch } from "react-redux";

const store = configureStore({
  reducer: {
    currentUser: currentUserReducer,
    train: trainReducer,
  },
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export const useAppDispatch: () => AppDispatch = useDispatch;
