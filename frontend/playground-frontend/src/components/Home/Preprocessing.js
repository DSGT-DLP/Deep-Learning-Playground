import TitleText from '../general/TitleText'
import CodeMirror from '@uiw/react-codemirror'
import { toast } from 'react-toastify'
import { python } from '@codemirror/lang-python'
import { basicSetup } from 'codemirror'
import Button from 'react-bootstrap/Button'
import React, { useState } from 'react'
import { userCodeEval } from '../helper_functions/TalkWithBackend'
import PropTypes from 'prop-types'

const Preprocessing = (props) => {
  const { data, setData, setColumns } = props
  const startingCode =
    'import pandas as pd\n\ndef preprocess(df): \n  # put your preprocessing code here!\n  return df'
  const [userCode, setUserCode] = useState(startingCode)
  const onChange = (value) => {
    setUserCode(value.target.innerText)
  }
  return (
    <div>
      <TitleText text='Preprocessing' />
      <CodeMirror
        value={startingCode}
        height='200px'
        extensions={[basicSetup, python()]}
        onBlur={onChange}
      />
      <Button
        onClick={async () => {
          const response = await userCodeEval(data, userCode)
          if (response['statusCode'] !== 200) {
            toast.error(response.message)
          } else {
            setData(response['data'])

            const newColumns = response['columns'].map((c) => ({
              name: c,
              selector: (row) => row[c],
            }))

            setColumns(newColumns)
            toast.success('Preprocessing successful!')
          }
        }}
      >
        Preprocess
      </Button>
    </div>
  )
}
Preprocessing.propTypes = {
  data: PropTypes.array.isRequired,
  setData: PropTypes.func.isRequired,
  setColumns: PropTypes.func.isRequired,
}

export default Preprocessing
