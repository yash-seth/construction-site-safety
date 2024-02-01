import React from 'react'
import WebcamImage from '../WebcamImage/WebcamImage'
import { useState } from 'react'
import CircularProgress from '@mui/material/CircularProgress';
import "./Worker.css"

function Worker({ webcamView, setWebcamView }) {
  const [file, setFile] = useState(null)
  const [resultLoading, setResultLoading] = useState(false)
  const [workerName, setWorkerName] = useState('')
  const [workerID, setWorkerID] = useState('')
  const [reportView, setReportView] = useState(false)
  const departments = ["------------------------","Architecture", "Brick Laying", "Painting"];
  const [department, setDepartment] = useState(departments[0]);
  
  const handleUpload = () => {
      if(!file) {
        alert("Image was not uploaded!")
        return
      }

      if(webcamView === 'open' && workerName === '') {
        alert('Worker Name is not entered!')
        return
      }

      if(workerID === '') {
        alert('Worker ID is not entered!')
        return
      }

      if(department === departments[0]) {
        alert('Worker Department is not selected!')
        return
      }
      
      setResultLoading(true)

      const fd = new FormData()
      fd.append('file', file)
      fd.append('filename', webcamView === 'open' ? workerName.split(' ')[0] + '_' + workerName.split(' ')[1] + '.jpg' : fd.get('file').name)
      fd.append('workerID', workerID)
      fd.append('workerDep', department)
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
                if(data.status === 'Error') {
                    alert('Please try again. There was an error!')
                  }

                document.getElementById("helmet-status-label").innerHTML = `${data['helmet_status'] ? 'Present' : 'Absent'}\n`
                document.getElementById("PPE-status-label").innerHTML = `${data['ppe_status'] ? 'Present' : 'Absent'}\n`
                document.getElementById("mask-status-label").innerHTML = `${data['mask_status'] ? 'Present' : 'Absent'}\n`
                setReportView(true)
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

  const exitReport = () => {
    setFile(null)
    document.getElementById("img-preview").src = null
    document.getElementById("img-preview").setAttribute("style", "display:none;")
    setWorkerName('')
    setWorkerID('')
    setDepartment(departments[0])
    setResultLoading(false)
    setReportView(false)
  }

  const handleWokerIDChange = (e) => {
    setWorkerID(e.target.value)
  }

  const handleDepartmentChange = (e) => {
    setDepartment(e.target.value)
  }

  return (
    <div className='worker-main'>
      <h1>Worker View</h1>
      {webcamView === 'open' && 
      <>
        <WebcamImage setWebcamView={setWebcamView} img={file} setImg={setFile} workerName={workerName} setWorkerName={setWorkerName} resultLoading={resultLoading} workerID={workerID} setWorkerID={setWorkerID} department={department} setDepartment={setDepartment} 
        departments={departments} />
      </>
      }
      {resultLoading === false && <label>Image: </label>}
      <img id="img-preview" alt='preview'></img>
      {resultLoading === false && <input type='file' onChange={ (e) => { handleImageSelect(e) } } onClick={() => setWebcamView('closed')}/>}

      {resultLoading === false && 
        <div className='image-controls'>
          {webcamView === 'closed' && 
            <>
              <label for='worker-id'>Enter ID: </label>
              <input onChange={ (e) => handleWokerIDChange(e) } value={workerID} id='worker-id' name='worker-id' type="text" />
              <label for='worker-department'>Select Department</label>
              <select
                onChange={(e) => handleDepartmentChange(e)}
                defaultValue={department}
                id="worker-department"
              >
                {departments.map((dep, idx) => (
                  <option key={idx}>{dep}</option>
                ))}
              </select>
              <button className="image-control-btn" onClick={() => {
              setFile(null);
              document.getElementById("img-preview").setAttribute("style", "display:none;");
              setWebcamView('open')
              }}>
                Capture Image
              </button>
            </>
          }
          {resultLoading ? <CircularProgress /> : <button className="image-control-btn" onClick={ handleUpload }>Upload</button>}
        </div>
      }

      <div>
      {resultLoading && 
        <>
          <div className='worker-report'>
            <h2>Worker Report</h2>
            <div className='worker-report-object'>
              <b>Helmet Status:</b>
              {reportView === false && <CircularProgress />}
              <label id="helmet-status-label"></label>
            </div>
            <div className='worker-report-object'>
              <b>PPE Status:</b>
              {reportView === false && <CircularProgress />}
              <label id="PPE-status-label"></label>
            </div>
            <div className='worker-report-object'>
            <b>Mask Status:</b>
              {reportView === false && <CircularProgress />}
              <label id="mask-status-label"></label>
            </div>
            <button id="worker-report-exit-btn" onClick={() => exitReport()}>Exit Report</button>
          </div>
        </>
        }
      </div>
    </div>
  )
}

export default Worker