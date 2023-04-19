import { PayloadAction, createSlice } from "@reduxjs/toolkit";
import { BaseTrainspaceData } from "../types/trainTypes";

interface TrainspaceState {
  current?: BaseTrainspaceData;
}
export const trainspaceSlice = createSlice({
  name: "trainspace",
  initialState: {} as TrainspaceState,
  reducers: {
    setTrainspace: (
      state,
      { payload }: PayloadAction<BaseTrainspaceData | undefined>
    ) => {
      state.current = payload;
    },
  },
});

export const { setTrainspace } = trainspaceSlice.actions;

export default trainspaceSlice.reducer;
