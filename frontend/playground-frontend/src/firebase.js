import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, GithubAuthProvider } from "firebase/auth";

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

export const googleProvider = new GoogleAuthProvider();
export const githubProvider = new GithubAuthProvider();
