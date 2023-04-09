import { PayloadAction, createAsyncThunk, createSlice } from "@reduxjs/toolkit";
//import { sendToBackend } from "../common/components/helper_functions/TalkWithBackend";
import { ThunkApiType } from "./store";
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

export const updateUserSettings = createAsyncThunk<
  UserType | undefined,
  {
    displayName?: string;
    email?: string;
    password?: string;
    checkPassword?: string;
  },
  ThunkApiType
>(
  "currentUser/updateUserSettings",
  async ({ displayName, email, password, checkPassword }, thunkAPI) => {
    if (password && !checkPassword) {
      return thunkAPI.rejectWithValue({
        message: "Please retype your password",
      });
    }
    if (!password && checkPassword) {
      return thunkAPI.rejectWithValue({
        message: "Please enter your password",
      });
    }
    if (password !== checkPassword) {
      return thunkAPI.rejectWithValue({
        message: "Passwords do not match",
      });
    }
    try {
      const user = thunkAPI.getState().currentUser.user;
      if (!auth.currentUser || !isSignedIn(user)) {
        const action = await thunkAPI.dispatch(signOutUser());
        if (action.meta.requestStatus === "rejected") {
          return thunkAPI.rejectWithValue(action.payload);
        }
        return;
      }
      if (displayName) {
        const action = await thunkAPI.dispatch(
          updateUserProfile({ displayName: displayName })
        );
        if (action.meta.requestStatus === "rejected") {
          return thunkAPI.rejectWithValue(action.payload);
        }
      }
      if (email) {
        const action = await thunkAPI.dispatch(
          updateUserEmail({ email: email })
        );
        if (action.meta.requestStatus === "rejected") {
          return thunkAPI.rejectWithValue(action.payload);
        }
      }
      if (password) {
        const action = await thunkAPI.dispatch(
          updateUserPassword({ password: password })
        );
        if (action.meta.requestStatus === "rejected") {
          return thunkAPI.rejectWithValue(action.payload);
        }
      }
      return {
        email: auth.currentUser.email,
        uid: auth.currentUser.uid,
        displayName: auth.currentUser.displayName,
        emailVerified: user.emailVerified,
      } as UserType;
    } catch (e) {
      if (e instanceof FirebaseError) {
        return thunkAPI.rejectWithValue({ message: e.message });
      }
    }
    return thunkAPI.rejectWithValue({
      message: "Update password failed",
    });
  }
);

export const updateUserPassword = createAsyncThunk<
  UserType | undefined,
  { password: string },
  ThunkApiType
>("currentUser/updateUserPassword", async ({ password }, thunkAPI) => {
  try {
    const user = thunkAPI.getState().currentUser.user;
    if (!auth.currentUser || !isSignedIn(user)) {
      const action = await thunkAPI.dispatch(signOutUser());
      if (action.meta.requestStatus === "rejected") {
        return thunkAPI.rejectWithValue(action.payload);
      }
      return;
    }
    await updatePassword(auth.currentUser, password);
    return {
      email: user.email,
      uid: user.uid,
      displayName: user.displayName,
      emailVerified: user.emailVerified,
    } as UserType;
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
  UserType | undefined,
  { email: string },
  ThunkApiType
>("currentUser/updateUserEmail", async ({ email }, thunkAPI) => {
  try {
    const user = thunkAPI.getState().currentUser.user;
    if (!auth.currentUser || !isSignedIn(user)) {
      const action = await thunkAPI.dispatch(signOutUser());
      if (action.meta.requestStatus === "rejected") {
        return thunkAPI.rejectWithValue(action.payload);
      }
      return;
    }
    await updateEmail(auth.currentUser, email);
    return {
      email: auth.currentUser.email,
      uid: user.uid,
      displayName: user.displayName,
      emailVerified: user.emailVerified,
    } as UserType;
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
  UserType | undefined,
  { displayName?: string; photoURL?: string },
  ThunkApiType
>(
  "currentUser/updateUserProfile",
  async ({ displayName, photoURL }, thunkAPI) => {
    try {
      const user = thunkAPI.getState().currentUser.user;
      if (!auth.currentUser || !isSignedIn(user)) {
        const action = await thunkAPI.dispatch(signOutUser());
        if (action.meta.requestStatus === "rejected") {
          return thunkAPI.rejectWithValue(action.payload);
        }
        return;
      }
      await updateProfile(auth.currentUser, {
        displayName: displayName,
        photoURL: photoURL,
      });
      return {
        email: user.email,
        uid: user.uid,
        displayName: auth.currentUser.displayName,
        emailVerified: user.emailVerified,
      } as UserType;
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
  UserType,
  {
    email: string;
    password: string;
    displayName: string;
    recaptcha: string | null;
  },
  ThunkApiType
>(
  "currentUser/registerViaEmailAndPassword",
  async ({ email, password, displayName, recaptcha }, thunkAPI) => {
    if (!recaptcha) {
      return thunkAPI.rejectWithValue({
        message: "Please complete the recaptcha",
      });
    }
    if (email === "") {
      return thunkAPI.rejectWithValue({
        message: "Please enter your email",
      });
    }
    if (password === "") {
      return thunkAPI.rejectWithValue({
        message: "Please enter your password",
      });
    }
    if (displayName === "") {
      return thunkAPI.rejectWithValue({
        message: "Please enter your display name",
      });
    }
    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        email,
        password
      );
      const user = userCredential.user;
      if (displayName) {
        await updateProfile(user, { displayName: displayName });
      }
      return {
        email: user.email,
        uid: user.uid,
        displayName: user.displayName,
        emailVerified: user.emailVerified,
      } as UserType;
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
  initialState: { user: "pending" } as UserState,
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
    builder.addCase(updateUserSettings.fulfilled, (state, { payload }) => {
      if (payload) {
        state.user = payload;
      }
    });
    builder.addCase(registerViaEmailAndPassword.pending, (state) => {
      state.user = "pending";
    });
    builder.addCase(
      registerViaEmailAndPassword.fulfilled,
      (state, { payload }) => {
        storage.setItem("expect-user", "true");
        state.user = payload;
      }
    );
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
