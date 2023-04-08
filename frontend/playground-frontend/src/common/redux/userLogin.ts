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
import { auth } from "../utils/firebase";
import { FirebaseError } from "firebase/app";

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

export const signInViaGoogle = createAsyncThunk<UserType, void, ThunkApiType>(
  "currentUser/signInViaGoogle",
  async (_, thunkAPI) => {
    try {
      const googleProvider = new GoogleAuthProvider();
      await signInWithRedirect(auth, googleProvider);
      const result = await getRedirectResult(auth);
      if (result) {
        if (!result.user.email || !result.user.displayName) {
          throw Error();
        }
        return {
          email: result.user.email,
          uid: result.user.uid,
          displayName: result.user.displayName,
          emailVerified: result.user.emailVerified,
        } as UserType;
      }
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue({ message: e.message });
      }
    }
    return thunkAPI.rejectWithValue({ message: "Sign in with Google failed" });
  }
);

export const signInViaGithub = createAsyncThunk<UserType, void, ThunkApiType>(
  "currentUser/signInViaGitHub",
  async (_, thunkAPI) => {
    try {
      const githubProvider = new GithubAuthProvider();
      await signInWithRedirect(auth, githubProvider);
      const result = await getRedirectResult(auth);
      if (result) {
        if (!result.user.email || !result.user.displayName) {
          throw Error();
        }
        return {
          email: result.user.email,
          uid: result.user.uid,
          displayName: result.user.displayName,
          emailVerified: result.user.emailVerified,
        } as UserType;
      }
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue({ message: e.message });
      }
    }
    return thunkAPI.rejectWithValue({ message: "Sign in with GitHub failed" });
  }
);

export const signOutUser = createAsyncThunk<void, void, ThunkApiType>(
  "currentUser/signOutUser",
  async (_, thunkAPI) => {
    try {
      await signOut(auth);
      return;
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue(Error(e.message));
      }
    }
    return thunkAPI.rejectWithValue({ message: "Sign out failed" });
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
      if (!payload && state.user) {
        state.user = undefined;
        state.userProgressData = undefined;
      } else if (payload && !state.user) {
        state.user = payload;
        if (payload.email && payload.displayName) {
          state.user = {
            email: payload.email,
            uid: payload.uid,
            displayName: payload.displayName,
            emailVerified: payload.emailVerified,
          };
        }
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(signOutUser.fulfilled, (state) => {
      state.user = undefined;
    });
    builder.addCase(signInViaGoogle.pending, (state) => {
      if (state.user) {
        state.user = undefined;
      }
    });
    builder.addCase(signInViaGoogle.fulfilled, (state, { payload }) => {
      if (state.user) {
        state.user = payload;
      }
    });
    builder.addCase(signInViaGithub.pending, (state) => {
      if (state.user) {
        state.user = undefined;
      }
    });
    builder.addCase(signInViaGithub.fulfilled, (state, { payload }) => {
      if (state.user) {
        state.user = payload;
      }
    });
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
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
