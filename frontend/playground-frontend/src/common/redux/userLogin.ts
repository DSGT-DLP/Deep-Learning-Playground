import { PayloadAction, createAsyncThunk, createSlice } from "@reduxjs/toolkit";
//import { sendToBackend } from "../common/components/helper_functions/TalkWithBackend";
import { AppDispatch, ThunkApiType } from "./store";
import {
  GithubAuthProvider,
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signInWithRedirect,
  signOut,
  updateEmail,
  updatePassword,
  updateProfile,
} from "firebase/auth";
import { auth } from "@/common/utils/firebase";
import { FirebaseError } from "firebase/app";
import storage from "local-storage-fallback";

export interface UserState {
  user?: UserType | "pending";
  userProgressData?: UserProgressDataType;
}

export interface UserType {
  email: string;
  uid: string;
  displayName?: string;
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

export const updateUserPassword = createAsyncThunk<
  void,
  { password: string },
  ThunkApiType
>("currentUser/updateUserPassword", async ({ password }, thunkAPI) => {
  try {
    if (!auth.currentUser) {
      await thunkAPI.dispatch(signOutUser());
      return;
    }
    await updatePassword(auth.currentUser, password);
    return;
  } catch (e) {
    if (e instanceof FirebaseError) {
      return thunkAPI.rejectWithValue({ message: e.message });
    }
  }
  return thunkAPI.rejectWithValue({
    message: "Update password failed",
  });
});

export const updateUserEmail = createAsyncThunk<
  void,
  { email: string },
  ThunkApiType
>("currentUser/updateUserEmail", async ({ email }, thunkAPI) => {
  try {
    if (!auth.currentUser) {
      await thunkAPI.dispatch(signOutUser());
      return;
    }
    await updateEmail(auth.currentUser, email);
    return;
  } catch (e) {
    if (e instanceof FirebaseError) {
      return thunkAPI.rejectWithValue({ message: e.message });
    }
  }
  return thunkAPI.rejectWithValue({
    message: "Update email failed",
  });
});

export const updateUserProfile = createAsyncThunk<
  void,
  { displayName?: string; photoURL?: string },
  ThunkApiType
>(
  "currentUser/updateUserProfile",
  async ({ displayName, photoURL }, thunkAPI) => {
    try {
      if (!auth.currentUser) {
        await thunkAPI.dispatch(signOutUser());
        return;
      }
      await updateProfile(auth.currentUser, {
        displayName: displayName,
        photoURL: photoURL,
      });
      return;
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue({ message: e.message });
      }
    }
    return thunkAPI.rejectWithValue({
      message: "Updating the user profile failed",
    });
  }
);

export const registerViaEmailAndPassword = createAsyncThunk<
  void,
  { email: string; password: string; displayName?: string },
  ThunkApiType
>(
  "currentUser/registerViaEmailAndPassword",
  async ({ email, password, displayName }, thunkAPI) => {
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCredential.user;
      await updateProfile(user, { displayName: displayName });
      storage.setItem("expect-user", "true");
      return;
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue({ message: e.message });
      }
    }
    return thunkAPI.rejectWithValue({
      message: "Registration with email and password failed",
    });
  }
);

export const signInViaEmailAndPassword = createAsyncThunk<
  void,
  { email: string; password: string },
  ThunkApiType
>(
  "currentUser/signInViaEmailAndPassword",
  async ({ email, password }, thunkAPI) => {
    try {
      await signInWithEmailAndPassword(auth, email, password);
      storage.setItem("expect-user", "true");
      return;
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue({ message: e.message });
      }
    }
    return thunkAPI.rejectWithValue({
      message: "Sign in with email and password failed",
    });
  }
);

export const signInViaGoogleRedirect = createAsyncThunk<
  void,
  void,
  ThunkApiType
>("currentUser/signInViaGoogle", async (_, thunkAPI) => {
  try {
    const googleProvider = new GoogleAuthProvider();
    storage.setItem("expect-user", "true");
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
    storage.setItem("expect-user", "true");
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
        storage.removeItem("expect-user");
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
      { payload }: PayloadAction<UserType | "pending" | undefined>
    ) => {
      if (!payload) {
        state.user = undefined;
        state.userProgressData = undefined;
      } else {
        state.user = payload;
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(signOutUser.fulfilled, (state) => {
      state.user = undefined;
    });
    builder.addCase(signInViaEmailAndPassword.pending, (state) => {
      state.user = "pending";
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
