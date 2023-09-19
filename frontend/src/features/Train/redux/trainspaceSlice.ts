import { PayloadAction, createAction, createSlice } from "@reduxjs/toolkit";
import { BaseTrainspaceData } from "@/features/Train/types/trainTypes";

interface TrainspaceState {
  current?: BaseTrainspaceData;
}

export const trainspaceSlice = createSlice({
  name: "trainspace",
  initialState: {} as TrainspaceState,
  reducers: {
    createTrainspaceData: (
      state,
      { payload }: PayloadAction<{ current: BaseTrainspaceData }>
    ) => {
      state.current = { ...payload.current, ...{ step: 0 } };
    },
    removeTrainspaceData: (state) => {
      state.current = undefined;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(updateTrainspaceData, (state, { payload }) => {
      state.current = { ...state.current, ...payload };
    });
  },
});

export const updateTrainspaceData = createAction<BaseTrainspaceData>(
  "trainspace/setTrainspaceData"
);

export const { createTrainspaceData, removeTrainspaceData } =
  trainspaceSlice.actions;

export default trainspaceSlice.reducer;
