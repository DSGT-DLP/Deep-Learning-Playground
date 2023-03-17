import React, { useState, useEffect } from 'react'
import storage from 'local-storage-fallback'
import Container from 'react-bootstrap/Container'
import DSGTLogo from '../../images/logos/dlp_branding/dlp-logo.png'
import Nav from 'react-bootstrap/Nav'
import NavDropdown from 'react-bootstrap/NavDropdown'
import Navbar from 'react-bootstrap/Navbar'
import { URLs } from '../../constants'
import { auth } from '../../firebase'
import { setCurrentUser } from '../../redux/userLogin'
import { signOut, onAuthStateChanged } from 'firebase/auth'
import { toast } from 'react-toastify'
import { useDispatch, useSelector } from 'react-redux'
import { useNavigate } from 'react-router-dom'
import { setCookie, deleteCookie } from '../helper_functions/Cookie'
import GlobalStyle from '../../GlobalStyle'
import { ThemeProvider } from 'styled-components'
import { FormControlLabel, Switch } from '@mui/material'

const NavbarMain = () => {
  const userEmail = useSelector((state) => state.currentUser.email)
  const dispatch = useDispatch()
  const navigate = useNavigate()

  function getInitialTheme() {
    const savedTheme = storage.getItem('theme')
    return savedTheme ? JSON.parse(savedTheme) : { mode: 'light', checked: false }
  }

  const [theme, setTheme] = useState(getInitialTheme)

  useEffect(() => {
    storage.setItem('theme', JSON.stringify(theme))
  }, [theme])

  const toggleTheme = () => {
    if (theme.mode === 'light') {
      setTheme({ mode: 'dark', checked: true })
    } else {
      setTheme({ mode: 'light', checked: false })
    }
  }

  const goToLogin = () => {
    if (!window.location.href.match(/\/login/g)) {
      // Go to Login page if we aren't already there
      navigate('/login')
    }
  }

  const logout = () => {
    signOut(auth)
      .then(() => {
        dispatch(setCurrentUser(null))
        toast.success('Logged out successfully', { autoClose: 1000 })
        deleteCookie('userEmail')
        navigate('/login')
      })
      .catch((error) => toast.error(`Error: ${error.code}`))
  }

  useEffect(() => {
    onAuthStateChanged(auth, (user) => {
      if (!user) return

      const userData = {
        email: user.email,
        uid: user.uid,
        displayName: user.displayName,
        emailVerified: user.emailVerified,
      }
      dispatch(setCurrentUser(userData))
      setCookie('userEmail', user.email)
    })
  }, [])

  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      <Navbar id='navbar-main' className='p-0' expand='lg'>
        <Container fluid className='ms-1 pe-0'>
          <Navbar.Brand href='/' className='d-flex align-items-center logo-title'>
            <img src={DSGTLogo} className='logo d-inline-block align-top me-3' alt='DSGT Logo' />
            Deep Learning Playground
          </Navbar.Brand>
          <Navbar.Toggle aria-controls='basic-navbar-nav' />
          <Navbar.Collapse id='basic-navbar-nav'>
            <Nav className='ms-auto'>
              {userEmail ? <Nav.Link href='/train'>Train</Nav.Link> : null}
              <Nav.Link href='/about'>About</Nav.Link>
              <Nav.Link href='/wiki'>Wiki</Nav.Link>
              <Nav.Link href='/feedback'>Feedback</Nav.Link>
              <Nav.Link href={URLs.donate}>Donate</Nav.Link>
              {userEmail ? (
                <NavDropdown title='Account' id='basic-nav-dropdown'>
                  <NavDropdown.Item href='/'>Dashboard</NavDropdown.Item>
                  <NavDropdown.Item href='/account-settings'>Settings</NavDropdown.Item>
                  <NavDropdown.Item href='learn-mod'>Learn</NavDropdown.Item>
                  <NavDropdown.Divider />
                  <NavDropdown.Item href='#' onClick={logout}>
                    Log out
                  </NavDropdown.Item>
                </NavDropdown>
              ) : (
                <Nav.Link href='#' onClick={goToLogin}>
                  Log in
                </Nav.Link>
              )}
            </Nav>
          </Navbar.Collapse>
        </Container>
        <FormControlLabel
          control={
            <Switch id='mode-switch' onChange={toggleTheme} checked={theme.checked}></Switch>
          }
          label={`${theme.mode === 'dark' ? '🌙' : '☀️'}`}
        />
      </Navbar>
    </ThemeProvider>
  )
}

export default NavbarMain
