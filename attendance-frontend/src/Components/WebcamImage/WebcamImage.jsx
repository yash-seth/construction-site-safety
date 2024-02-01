import Webcam from "react-webcam";
import React, { useRef, useCallback } from "react";
import "./WebcamImage.css"

// get workers name for the file name

function WebcamImage({ setWebcamView, img, setImg, workerName, setWorkerName, resultLoading, setWorkerID, workerID }) {
  const webcamRef = useRef(null);

    const capture = useCallback(() => {
        const imageSrc = webcamRef.current.getScreenshot();
        // console.log(imageSrc)
        setImg(imageSrc);
    }, [webcamRef]);

    const videoConstraints = {
        width: 390,
        height: 390,
        facingMode: "user",
    };
    
    const handleWokerNameChange = (e) => {
      setWorkerName(e.target.value)
    }

    const handleWokerIDChange = (e) => {
      setWorkerID(e.target.value)
    }

  return (
    <div className="WebcamImage-main">
      {resultLoading === false && <label for='worker-name'>Enter Name: </label>}
      <input onChange={ (e) => handleWokerNameChange(e) } value={workerName} id='worker-name' name='worker-name' type="text" />
      {resultLoading === false && <label for='worker-id'>Enter ID: </label>}
      <input onChange={ (e) => handleWokerIDChange(e) } value={workerID} id='worker-id' name='worker-id' type="text" />
      {img === null ? (
        <>
          <Webcam
            screenshotFormat="image/jpeg"
            videoConstraints={videoConstraints}
            audio={false}
            height={500}
            width={500}
            ref={webcamRef}
            mirrored={true}
          />
          {resultLoading === false && <button className="image-control-btn" onClick={capture}>Capture photo</button>}
        </>
      ) : (
        <>
          <img src={img} alt="screenshot" />
          {resultLoading === false && <button className="image-control-btn" onClick={() => setImg(null)}>Recapture</button>}
        </>
      )}
      {resultLoading === false && <button className="image-control-btn" onClick={() => {setWebcamView('closed'); setImg(null); setWorkerName('')}}>Close</button>}
    </div>
  );
}

export default WebcamImage;