import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { User } from "firebase/auth";
import { sendToBackend } from "../components/helper_functions/TalkWithBackend";
import { ModuleType } from "../components/LearnMod/LearningModulesContent";
import { ThunkApiType } from "./store";

export interface UserState {
  email?: string;
  uid?: string;
  displayName?: string;
  emailVerified?: boolean;
  userProgressData?: UserProgressDataType;
}

interface UserProgressDataType {
  progressData: string;
}

export const fetchUserProgressData = createAsyncThunk<
  UserProgressDataType,
  void,
  ThunkApiType
>("currentUser/fetchUserProgressData", async (_, thunkAPI) => {
  if (thunkAPI.getState().currentUser.uid) {
    return await sendToBackend(
      "getUserProgressData",
      thunkAPI.getState().currentUser.uid
    );
  }
  return thunkAPI.rejectWithValue(
    Error("User does not exist in currentUser slice")
  );
});

export const currentUserSlice = createSlice({
  name: "currentUser",
  initialState: { userDataFetchInitiated: false } as UserState,
  reducers: {
    setCurrentUser: (state, { payload }: { payload: UserState }) => {
      if (!payload) {
        state.email = undefined;
        state.uid = undefined;
        state.displayName = undefined;
        state.emailVerified = undefined;
        state.userProgressData = undefined;
        return;
      }

      const { email, uid, displayName, emailVerified } = payload;
      state.email = email;
      state.uid = uid;
      state.displayName = displayName;
      state.emailVerified = emailVerified;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchUserProgressData.pending, (state) => {
      state.userProgressData = undefined;
    });
    builder.addCase(fetchUserProgressData.fulfilled, (state, { payload }) => {
      state.userProgressData = payload;
    });
    builder.addCase(fetchUserProgressData.rejected, (state, { payload }) => {
      state.userProgressData = undefined;
      console.log(payload);
    });
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
