import { configureStore } from "@reduxjs/toolkit";
import currentUserReducer from "./userLogin";
import trainReducer from "./train";
import trainspaceReducer from "@/features/Train/redux/trainspaceSlice";
import { backendApi } from "./backendApi";

const store = configureStore({
  reducer: {
    currentUser: currentUserReducer,
    train: trainReducer,
    trainspace: trainspaceReducer,
    [backendApi.reducerPath]: backendApi.reducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(backendApi.middleware),
  devTools: true,
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export interface ThunkApiType {
  dispatch: AppDispatch;
  state: RootState;
}
