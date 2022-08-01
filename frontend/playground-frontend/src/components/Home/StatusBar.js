// import React, { useState, useEffect } from 'react'
import React from 'react'
import PropTypes from "prop-types"
import { Circle } from 'rc-progress'

const StatusBar = ({ pendingResponse, setPendingResponse, progress }) => {
  console.log('log:', progress)

  // const [logData, setLogData] = useState(null)
  // const [stream, setStream] = useState(null)
  
  // console.log('pendingResponse in StatusBar:', pendingResponse)

  // useEffect(() => {
  //   const getData = async () => {
  //     if (pendingResponse) {
  //       // console.log('waiting')
  //       // const logResponse = await fetch('/training_log').then((res) => res.json()).then((data) => data.log)
  //       // console.log('logResponse', logResponse)
  //       // if (logResponse && logResponse !== logData) {
  //       //   // console.log(logData)
  //       //   setLogData(logResponse)
  //       // }
  //     }
  //   }
  //   getData()
  // }, )
// 
  // useEffect(() => {
  //   if (!pendingResponse) {
  //     setLogData(null)
  //   }
  // }, [pendingResponse])

  // console.log('logData:', logData)

  if (progress) {
    return <div style={{ margin: 10, width: 200 }}><Circle percent={progress} strokeWidth={4} /></div>
  } else {
    return <></>
  }
}

StatusBar.propTypes = {
  pendingResponse: PropTypes.bool.isRequired,
  setPendingResponse: PropTypes.func.isRequired,
  progress: PropTypes.number
}

export default StatusBar