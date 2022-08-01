// import React, { useState, useEffect } from 'react'
import React, { useEffect, useState } from 'react'
import PropTypes from "prop-types";
import { io } from 'socket.io-client'

const socket = io('http://localhost:5000')
socket.on('connect', () => {
  console.log(socket)
})
socket.on('connect_error', (err) => {
  console.log(`connect_error due to ${err.message}`)
  socket.close()
})

const StatusBar = ({ pendingResponse, setPendingResponse }) => {
  const [logData, setLogData] = useState()

  useEffect(() => {
    console.log('inside', socket)
    socket.on('SendLog', (log) => {
      // console.log(log)
      setLogData(log)
    })
  }, [socket])

  useEffect(() => {
    console.log('in useEffect', socket)
  }, [pendingResponse])

  console.log('logData', logData)

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

  return <></>
}

StatusBar.propTypes = {
  pendingResponse: PropTypes.bool.isRequired,
  setPendingResponse: PropTypes.func.isRequired
}

export default StatusBar