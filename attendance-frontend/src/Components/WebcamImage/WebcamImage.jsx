import Webcam from "react-webcam";
import React, { useRef, useCallback } from "react";
import "./WebcamImage.css"

// get workers name for the file name

function WebcamImage({ setWebcamView, img, setImg }) {
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

  return (
    <div className="Container">
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
          <button onClick={capture}>Capture photo</button>
        </>
      ) : (
        <>
          <img src={img} alt="screenshot" />
          <button onClick={() => setImg(null)}>Recapture</button>
        </>
      )}
      <button onClick={() => {setWebcamView('closed'); setImg(null)}}>Close</button>
    </div>
  );
}

export default WebcamImage;