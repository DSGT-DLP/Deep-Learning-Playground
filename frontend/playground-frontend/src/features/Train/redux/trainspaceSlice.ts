import { PayloadAction, createSlice } from "@reduxjs/toolkit";
import { BaseTrainspaceData } from "../types/trainTypes";

interface TrainspaceState {
  current?: BaseTrainspaceData;
}
export const trainspaceSlice = createSlice({
  name: "trainspace",
  initialState: {} as TrainspaceState,
  reducers: {
    setTrainspaceData: (
      state,
      { payload }: PayloadAction<BaseTrainspaceData | undefined>
    ) => {
      if (!payload) {
        state.current = undefined;
      }
      if (!state.current) {
        state.current = payload;
        return;
      }
      state.current = { ...state.current, ...payload };
    },
  },
});

export const { setTrainspaceData } = trainspaceSlice.actions;

export default trainspaceSlice.reducer;
