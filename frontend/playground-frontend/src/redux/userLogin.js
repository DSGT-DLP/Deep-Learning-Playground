import { createSlice } from "@reduxjs/toolkit";

export const currentUserSlice = createSlice({
  name: "currentUser",
  initialState: {
    email: null,
    uid: null,
  },
  reducers: {
    setCurrentUser: (state, action) => {
      state.email = action.payload.email;
      state.uid = action.payload.uid;
      console.log(3243, action.payload);
    },
  },
});

export const { setCurrentUser } = currentUserSlice.actions;

export default currentUserSlice.reducer;
