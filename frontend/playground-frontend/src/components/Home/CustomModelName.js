import React from 'react'
import { Form } from 'react-bootstrap'
import { useDispatch, useSelector } from 'react-redux'
import { setModelName } from '../../redux/train'

const CustomModelName = () => {
  const customModelName = useSelector((state) => state.train.customModelName)
  const dispatch = useDispatch()
  return (
    <>
      <Form.Control
        className='model-name-input'
        placeholder='Give a custom model name'
        value={customModelName}
        onChange={(e) => dispatch(setModelName(e.target.value))}
        maxLength={255}
      />
    </>
  )
}

export default CustomModelName
