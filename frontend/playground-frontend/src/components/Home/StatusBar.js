import React, { useState, useEffect } from 'react'
import PropTypes from "prop-types";

const StatusBar = ({ pendingResponse, setPendingResponse }) => {
  const [logData, setLogData] = useState(null)
  const [stream, setStream] = useState(null)
  
  console.log('pendingResponse in StatusBar:', pendingResponse)

  useEffect(() => {
    console.log('stream', stream)
    if (pendingResponse) {
      let logStream
      if (!stream) {
        logStream = new EventSource('/training_log')
        setStream(logStream)
      } else {
        logStream = stream
      }
      logStream.onmessage = (event) => {
        console.log('handle stream:', event)
        setLogData(event.data)
      }
      logStream.onerror = (error) => {
        console.log('error:', error)
        console.log('closing stream in error')
        logStream.close()
      }
    } else {
      if (stream) {
        stream.close()
        console.log('closing stream in else statement')
        setLogData(null)
        setStream(null)
      }
    }
  }, )

  console.log('logData:', logData)

  return <></>
}

StatusBar.propTypes = {
  pendingResponse: PropTypes.bool.isRequired,
  setPendingResponse: PropTypes.func.isRequired
}

export default StatusBar