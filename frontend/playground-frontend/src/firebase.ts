import { initializeApp } from "firebase/app";
import {
  getAuth,
  GoogleAuthProvider,
  GithubAuthProvider,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithRedirect,
  updateProfile,
  updateEmail,
  updatePassword,
} from "firebase/auth";
import { toast } from "react-toastify";
import { setCookie } from "./components/helper_functions/Cookie";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAMJgYSG_TW7CT_krdWaFUBLxU4yRINxX8",
  authDomain: "deep-learning-playground-8d2ce.firebaseapp.com",
  projectId: "deep-learning-playground-8d2ce",
  storageBucket: "deep-learning-playground-8d2ce.appspot.com",
  messagingSenderId: "771338023154",
  appId: "1:771338023154:web:8ab6e73fc9c646426a606b",
};

// Initialize Firebase
export const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// Exported functions

export const updateUserProfile = async (
  displayName: string | null = null,
  photoURL: string | null = null
) => {
  const newDetails: {
    displayName?: string;
    photoURL?: string;
  } = {};
  if (displayName != null) newDetails.displayName = displayName;
  if (photoURL != null) newDetails.photoURL = photoURL;
  if (!auth?.currentUser)
    throw new Error("Firebase Auth current user is missing");

  await updateProfile(auth.currentUser, newDetails).catch((e) =>
    toast.error(`Error: ${e.code}`, { autoClose: 1000 })
  );
};

export const updateUserSettings = async (
  displayName: string | null = null,
  email: string,
  password: string
) => {
  if (!auth?.currentUser)
    throw new Error("Firebase Auth current user is missing");

  updateEmail(auth.currentUser, email)
    .then(() => {
      const user = auth.currentUser;
      if (!user) throw new Error("Firebase Auth updated user is missing");

      updateUserProfile(displayName);
      setCookie("email", user.email);
      toast.success(`Updated email to ${user.email}`, {
        autoClose: 1000,
      });
      updatePassword(user, password)
        .then(() => {
          toast.success("Updated Password", {
            autoClose: 1000,
          });
        })
        .catch((e) => toast.error(`Error: ${e.code}`, { autoClose: 1000 }));
      return user;
    })
    .catch((e) => toast.error(`Error: ${e.code}`, { autoClose: 1000 }));
};

export const registerWithPassword = async (
  email: string,
  password: string,
  displayName: string | null = null
) => {
  return await createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      updateUserProfile(displayName);
      setCookie("email", user.email);
      toast.success(`Registered with email ${user.email}`, {
        autoClose: 1000,
      });
      return user;
    })
    .catch((error) => {
      toast.error(`Error: ${error.code}`);
      return false;
    });
};

export const signInWithPassword = async (email: string, password: string) => {
  return signInWithEmailAndPassword(auth, email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      setCookie("email", user.email);
      toast.success(`Signed in with email ${user.email}`, {
        autoClose: 1000,
      });
      return user;
    })
    .catch((error) => {
      toast.error(`Error: ${error.code}`);
      return false;
    });
};

export const signInWithGithub = async () => {
  const githubProvider = new GithubAuthProvider();
  signInWithRedirect(auth, githubProvider);
};

export const signInWithGoogle = async () => {
  const googleProvider = new GoogleAuthProvider();
  signInWithRedirect(auth, googleProvider);
};
