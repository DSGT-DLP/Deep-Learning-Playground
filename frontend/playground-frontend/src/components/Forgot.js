// import React, {useState} from "react";
// // import Form from "react-bootstrap/Form";
// import {Link} from 'react-router-dom';
// import Form from "react-bootstrap/Form";
// import {getAuth, sendPasswordResetEmail} from 'firebase/auth';
// import {toast} from 'react-toastify';
// // impor../../firebaseponent as ArrowRightIcon} from '../assets/svg'

// function Forgot() {
//   const [email, setEmail] = useState('');

//   const onChange = e => setEmail(e.target.value);

//   const onSubmit = async e => {
//     e.preventDefault();
//     try {
//       const auth = getAuth();
//       await sendPasswordResetEmail(auth, email);
//       toast.success('Email was sent');
//     } catch (error) {
//       toast.error('Could not send reset email');
//     }
//   };

// return (
//   <div id="forgot-page" className="text-center">
//     <div className="main-container mt-5 mb-5 mb-3 justify-content-evenly">
//       <h2 className="title">Reset Password</h2>
//       <Form.Label>Email address</Form.Label>
//       <form onSubmit={onSubmit}>
//         <input type='email' 
//         className='emailInput'
//         placeholder='someone@example.com'
//         id='email'
//         value={email}
//         onChange={onChange}
//         />
//         <div className="d-flex flex-column">
//           <button className="buttons">
//             Send Reset Link
//           </button>
//         </div>
//         <div>
//         <Link className="forgetPasswordlink" to='/login'>
//           Sign In
//         </Link>
//         </div>
//       </form>
//   </div>
//   </div>
// );
// }

// export default Forgot;
