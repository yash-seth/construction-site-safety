import React from 'react'
import "./Navbar.css"

function Navbar({ setViewState }) {
  return (
    <div className='NavbarMain'>
        <button id='navbar-controls-btn' onClick={() => setViewState('admin')}>Admin View</button>
        <button id='navbar-controls-btn' onClick={() => setViewState('worker')}>Worker View</button>
    </div>
  )
}

export default Navbar