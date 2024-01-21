import React from 'react'
import { useState } from 'react'
import "./Worker.css"

function Worker() {
  const [file, setFile] = useState()
  // const [status, setStatus] = useState(false)
  const handleUpload = () => {
      if(!file) {
        alert("Image was not uploaded!")
        return
      }

      const fd = new FormData()
      fd.append('file', file)
      fd.append('filename', fd.get('file').name)
      console.log(fd.get('file'))

      fetch("http://127.0.0.1:5000/upload", {
            method: 'POST',
            headers: {
              'Accept': 'application/json'
            },
            body: fd})
        .then((res) =>
            res.json().then((data) => {
                console.log(data)
                // setStatus(data.status)
                if(data.status === 'True') alert('Helmet detected!')
                else if(data.status === 'False') alert('Helmet not detected!')
            })
        )
        .catch((err) => console.log(err));
  }
  
  return (
    <div className='worker-main'>
      <h1>Worker View</h1>
      <label>Image: </label>
      <input type='file' onChange={ (e) => { setFile(e.target.files[0]) } }/>
      <button onClick={ handleUpload }>Upload</button>
    </div>
  )
}

export default Worker