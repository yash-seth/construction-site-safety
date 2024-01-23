import React from 'react'
import WebcamImage from '../WebcamImage/WebcamImage'
import { useState } from 'react'
import CircularProgress from '@mui/material/CircularProgress';
import "./Worker.css"

function Worker({ webcamView, setWebcamView }) {
  const [file, setFile] = useState(null)
  const [resultLoading, setResultLoading] = useState(false)
  
  const handleUpload = () => {
      if(!file) {
        alert("Image was not uploaded!")
        return
      }
      
      setResultLoading(true)

      const fd = new FormData()
      fd.append('file', file)
      fd.append('filename', webcamView === 'open' ? 'yash_seth.jpg' : fd.get('file').name)
      fd.append('source', webcamView === 'open' ? 'capture' : 'upload')
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
                else if(data.status === 'Error') {
                  alert('Please try again. There was an error!')
                }
                setFile(null)
                document.getElementById("img-preview").src = null
                document.getElementById("img-preview").setAttribute("style", "display:none;")
                setResultLoading(false)
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
      {webcamView === 'open' && 
      <>
        <WebcamImage setWebcamView={setWebcamView} img={file} setImg={setFile}/>
      </>
      }
      <label>Image: </label>
      <img id="img-preview" alt='preview'></img>
      <input type='file' onChange={ (e) => { handleImageSelect(e) } } onClick={() => setWebcamView('closed')}/>
      {webcamView === 'closed' && 
        <button onClick={() => {
          setFile(null);
          document.getElementById("img-preview").setAttribute("style", "display:none;");
          setWebcamView('open')
          }}>
            Capture Image
        </button>}
      {resultLoading ? <CircularProgress /> : <button onClick={ handleUpload }>Upload</button>}
    </div>
  )
}

export default Worker