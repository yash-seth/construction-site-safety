import './App.css';
import { useState } from 'react'
import Navbar from './Components/Navbar/Navbar';
import Admin from './Components/Admin/Admin';
import Worker from './Components/Worker/Worker';

function App() {
  const [viewState, setViewState] = useState('admin')
  return (
    <div>
      <Navbar setViewState={setViewState}/>
      {viewState === 'admin' && <Admin />}
        {viewState === 'worker' && <Worker />}
    </div>
  );
}

export default App;
