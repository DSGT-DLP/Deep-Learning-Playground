import { createSlice } from "@reduxjs/toolkit";

interface UserState {
  email?: string;
  uid?: string;
  displayName?: string;
  emailVerified?: boolean;
}

export const currentUserSlice = createSlice({
  name: "currentUser",
  initialState: {} as UserState,
  reducers: {
    setCurrentUser: (state, action) => {
      const payload = action.payload;
      if (!payload) {
        state.email = undefined;
        state.uid = undefined;
        state.displayName = undefined;
        state.emailVerified = undefined;
        return;
      }

      const { email, uid, displayName, emailVerified } = payload;
      state.email = email;
      state.uid = uid;
      state.displayName = displayName;
      state.emailVerified = emailVerified;
    },
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
