import React, { useState } from 'react'
import PropTypes from 'prop-types'
import { Form } from 'react-bootstrap'
import { GENERAL_STYLES } from '../../constants'

const PhoneNumberInput = (props) => {
  const { setPhoneNumber } = props
  const [numberIsValid, setNumberIsValid] = useState(true)

  function updateNumberInput(numberInput) {
    if (!validateNumber(numberInput)) {
      setNumberIsValid(false)
      return
    }
    setNumberIsValid(true)
    setPhoneNumber(numberInput)
  }

  return (
    <Form>
      <Form.Control
        style={{ width: '25%' }}
        maxLength={64}
        placeholder='+16785552057'
        onBlur={(e) => updateNumberInput(e.target.value)}
      />
      {numberIsValid ? null : (
        <p style={GENERAL_STYLES.error_text}>
          Please enter a valid number with the country code e.g. +16785553058
        </p>
      )}
    </Form>
  )
}

/**
 * Validates the phone number according to format +238201934012080123
 * @param {string} phoneNumber
 * @returns true if valid, false if not
 */
export function validateNumber(phoneNumber) {
  if (phoneNumber && !phoneNumber.match(/^\+\d*$/)) {
    return false
  }
  return true
}

PhoneNumberInput.propTypes = {
  setPhoneNumber: PropTypes.func.isRequired,
}

export default PhoneNumberInput
