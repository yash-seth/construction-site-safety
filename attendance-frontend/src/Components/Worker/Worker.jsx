import React from 'react'
import { useState } from 'react'
import "./Worker.css"

function Worker() {
  const [file, setFile] = useState()
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
                if(data.status === 'True') alert('Helmet detected!')
                else if(data.status === 'False') alert('Helmet not detected!')
            })
        )
        .catch((err) => console.log(err));
  }
  
  const handleImageSelect = (e) => {
    setFile(e.target.files[0])

    if (e.target.files[0]) {
      let imgURL = URL.createObjectURL(e.target.files[0]);
      document.getElementById("img-preview").src = imgURL
      document.getElementById("img-preview").setAttribute("style", "display:block;")
    }
  }

  return (
    <div className='worker-main'>
      <h1>Worker View</h1>
      <label>Image: </label>
      <img id="img-preview" alt='preview'></img>
      <input type='file' onChange={ (e) => { handleImageSelect(e) } }/>
      <button onClick={ handleUpload }>Upload</button>
    </div>
  )
}

export default Worker