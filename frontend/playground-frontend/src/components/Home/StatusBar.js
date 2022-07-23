import React, { useState, useEffect } from 'react'
import PropTypes from "prop-types";

const StatusBar = ({ pendingResponse, setPendingResponse }) => {
  const [logData, setLogData] = useState(null)
  // const [stream, setStream] = useState(null)
  
  console.log('pendingResponse in StatusBar:', pendingResponse)

  useEffect(() => {
    let logStream
    if (pendingResponse) {
      logStream = new EventSource('/training_log')
      
      logStream.onmessage = (event) => {
        console.log('handle stream:', event)
        setLogData(event.data)
      }
    } else {
      if (logStream) {
        logStream.close()
        console.log('closing stream in else statement')
        setLogData(null)
      }
      setLogData(null)
    }
    return () => {
      if (logStream) {
        logStream.close()
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