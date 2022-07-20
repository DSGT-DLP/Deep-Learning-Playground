import { useState } from "react";
import { useField } from "formik";

export const TextField = ({ label, disabled, ...props }) => {
  const [field, meta, helper] = useField(props);

  const error = meta.error;
  const showErrorMessage = meta.touched && error;

  return (
    <div className="textfield">
      <label htmlFor={props.id || props.name}>{label}</label>
      <div>
        <input
          {...field}
          type={props.type}
          placeholder={props.placeholder}
          disabled={disabled ? true : false}
        />
      </div>
      <p>{error && showErrorMessage}</p>
    </div>
  );
};

export const ProtectedField = (props) => {
  const [show, setShow] = useState(false);

  return (
    <TextField {...props} type={show ? "text" : "password"}>
      <div>
        <span>Show</span>
        <button
          type="button"
          onClick={() => setShow((prev) => !prev)}
          disabled={props.disabled}
        >
          {show && <p>check</p>}
        </button>
      </div>
    </TextField>
  );
};
