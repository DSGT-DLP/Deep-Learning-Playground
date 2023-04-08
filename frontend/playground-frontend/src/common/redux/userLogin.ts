import { PayloadAction, createAsyncThunk, createSlice } from "@reduxjs/toolkit";
//import { sendToBackend } from "../common/components/helper_functions/TalkWithBackend";
import { ThunkApiType } from "./store";
import {
  GithubAuthProvider,
  GoogleAuthProvider,
  User,
  getRedirectResult,
  signInWithRedirect,
  signOut,
} from "firebase/auth";
import { auth } from "@/common/utils/firebase";
import { FirebaseError } from "firebase/app";

export interface UserState {
  user?: UserType | "pending";
  userProgressData?: UserProgressDataType;
}

export interface UserType {
  email: string;
  uid: string;
  displayName: string;
  emailVerified: boolean;
}
export const isSignedIn = (
  user: UserType | "pending" | undefined
): user is UserType => user != undefined && user != "pending";

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

export const signInViaGoogleRedirect = createAsyncThunk<
  void,
  void,
  ThunkApiType
>("currentUser/signInViaGoogle", async (_, thunkAPI) => {
  try {
    const googleProvider = new GoogleAuthProvider();
    await signInWithRedirect(auth, googleProvider);
    return;
  } catch (e) {
    if (e instanceof FirebaseError) {
      return thunkAPI.rejectWithValue({ message: e.message });
    }
  }
  return thunkAPI.rejectWithValue({ message: "Sign in with Google failed" });
});

export const signInViaGithubRedirect = createAsyncThunk<
  void,
  void,
  ThunkApiType
>("currentUser/signInViaGitHub", async (_, thunkAPI) => {
  try {
    const githubProvider = new GithubAuthProvider();
    await signInWithRedirect(auth, githubProvider);
    return;
  } catch (e) {
    if (e instanceof FirebaseError) {
      return thunkAPI.rejectWithValue({ message: e.message });
    }
  }
  return thunkAPI.rejectWithValue({ message: "Sign in with GitHub failed" });
});

export const signOutUser = createAsyncThunk<void, void, ThunkApiType>(
  "currentUser/signOutUser",
  async (_, thunkAPI) => {
    if (thunkAPI.getState().currentUser.user) {
      try {
        await signOut(auth);
        return;
      } catch (e) {
        if (e instanceof FirebaseError) {
          return thunkAPI.rejectWithValue({ message: e.message });
        }
      }
      return thunkAPI.rejectWithValue({ message: "Sign out failed" });
    }
  }
);

export const fetchUserProgressData = createAsyncThunk<
  UserProgressDataType,
  void,
  ThunkApiType
>("currentUser/fetchUserProgressData", async (_, thunkAPI) => {
  if (thunkAPI.getState().currentUser.user) {
    //const result: string = await sendToBackend("getUserProgressData", {});
    //return JSON.parse(result);
  }
  return thunkAPI.rejectWithValue({
    message: "User does not exist in currentUser slice",
  });
});

export const currentUserSlice = createSlice({
  name: "currentUser",
  initialState: {} as UserState,
  reducers: {
    setCurrentUser: (
      state,
      { payload }: PayloadAction<UserType | undefined>
    ) => {
      if (!payload) {
        state.user = undefined;
        state.userProgressData = undefined;
      } else if (payload && !state.user) {
        state.user = payload;
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(signOutUser.fulfilled, (state) => {
      state.user = undefined;
    });
    builder.addCase(signInViaGoogleRedirect.pending, (state) => {
      state.user = "pending";
    });
    builder.addCase(signInViaGithubRedirect.pending, (state) => {
      state.user = "pending";
    });
    builder.addCase(fetchUserProgressData.pending, (state) => {
      state.userProgressData = undefined;
    });
    builder.addCase(fetchUserProgressData.fulfilled, (state, { payload }) => {
      state.userProgressData = payload;
    });
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
