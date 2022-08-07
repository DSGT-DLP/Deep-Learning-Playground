import { initializeApp } from "firebase/app";
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCMq0RmoCB1G6JHwwAvVXdcaykAhvwfv4Q",
  authDomain: "dsgt-authentication.firebaseapp.com",
  projectId: "dsgt-authentication",
  storageBucket: "dsgt-authentication.appspot.com",
  messagingSenderId: "15938712141",
  appId: "1:15938712141:web:0ecd946393c0584e72a204",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);