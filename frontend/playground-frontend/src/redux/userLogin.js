import { createSlice } from "@reduxjs/toolkit";

export const currentUserSlice = createSlice({
  name: "currentUser",
  initialState: {
    email: null,
    uid: null,
    displayName: null,
    emailVerified: null,
  },
  reducers: {
    setCurrentUser: (state, action) => {
      const payload = action.payload;
      if (!payload) {
        state.email = null;
        state.uid = null;
        state.displayName = null;
        state.emailVerified = null;
        return;
      }

      const { email, uid, displayName, emailVerified } = payload;
      state.email = email ?? null;
      state.uid = uid ?? null;
      state.displayName = displayName ?? null;
      state.emailVerified = emailVerified ?? null;
    },
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
