import './App.css';
import { useState } from 'react'
import Navbar from './Components/Navbar/Navbar';
import Admin from './Components/Admin/Admin';
import Worker from './Components/Worker/Worker';

function App() {
  const [viewState, setViewState] = useState('worker')
  const [webcamView, setWebcamView] = useState('closed')
  return (
    <div>
      <Navbar setViewState={setViewState}/>
      {viewState === 'admin' && <Admin />}
        {viewState === 'worker' && <Worker webcamView={webcamView} setWebcamView={setWebcamView}/>}
    </div>
  );
}

export default App;
