import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { sendToBackend } from "../components/helper_functions/TalkWithBackend";
import { ThunkApiType } from "./store";

export interface UserState {
  user?: UserType;
  userProgressData?: UserProgressDataType;
}

export interface UserType {
  email: string;
  uid: string;
  displayName: string;
  emailVerified: boolean;
}
interface UserProgressDataType {
  modules: {
    [moduleID: string]: {
      modulePoints: number;
      sections: {
        [sectionID: string]: {
          sectionPoints: number;
          questions: { [questionID: string]: number };
        };
      };
    };
  };
}

export const fetchUserProgressData = createAsyncThunk<
  UserProgressDataType,
  void,
  ThunkApiType
>("currentUser/fetchUserProgressData", async (_, thunkAPI) => {
  if (thunkAPI.getState().currentUser.user) {
    const result: string = await sendToBackend("getUserProgressData", {});
    return JSON.parse(result);
  }
  return thunkAPI.rejectWithValue(
    Error("User does not exist in currentUser slice")
  );
});

export const currentUserSlice = createSlice({
  name: "currentUser",
  initialState: {} as UserState,
  reducers: {
    setCurrentUser: (state, { payload }: { payload: UserType | undefined }) => {
      if (!payload) {
        state.user = undefined;
        state.userProgressData = undefined;
      }
      state.user = payload;
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchUserProgressData.pending, (state) => {
      if (state.user) {
        state.userProgressData = undefined;
      }
    });
    builder.addCase(fetchUserProgressData.fulfilled, (state, { payload }) => {
      if (state.user) {
        state.userProgressData = payload;
      }
    });
    builder.addCase(fetchUserProgressData.rejected, (state, { payload }) => {
      if (state.user) {
        state.userProgressData = undefined;
        console.log(payload);
      }
    });
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
