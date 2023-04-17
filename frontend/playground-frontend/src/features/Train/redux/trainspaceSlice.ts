import { PayloadAction, createSlice } from "@reduxjs/toolkit";
import { BaseTrainspaceData } from "../types/trainTypes";

export const trainspaceSlice = createSlice({
  name: "trainspace",
  initialState: null as BaseTrainspaceData | null,
  reducers: {
    setTrainspace: (state, { payload }: PayloadAction<BaseTrainspaceData>) => {
      state = payload;
    },
  },
});

export const { setTrainspace } = trainspaceSlice.actions;

export default trainspaceSlice.reducer;
