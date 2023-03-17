import { configureStore } from '@reduxjs/toolkit'
import currentUserReducer from './userLogin'
import trainReducer from './train'

const store = configureStore({
  reducer: {
    currentUser: currentUserReducer,
    train: trainReducer,
  },
})

export default store

export type RootState = ReturnType<typeof store.getState>
