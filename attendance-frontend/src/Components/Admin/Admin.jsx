import React from 'react'
import { useState } from 'react'
import "./Admin.css"
import CircularProgress from '@mui/material/CircularProgress';
import WorkerLogs from './worker_log.csv';

function Admin() {
    const [query, setQuery] = useState('')
    const [answer, setAnswer] = useState({
        response: ''
    })

    const [syncLoading, setSyncLoading] = useState(false)
    const [currentDate, setCurrentDate] = useState()

    const handleChange = (event) => {
        const value = event.target.value;
        setQuery(value);
    };

    const queryLLM = () => {
        fetch("http://127.0.0.1:5000/query", {
            method: 'POST',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({query: query})})
        .then((res) =>
            res.json().then((data) => {
                setAnswer({
                    response: data.response
                });
            })
        );
    }

    const syncLogs = () => {
        setSyncLoading(true)
        fetch("http://127.0.0.1:5000/syncLogs", {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }})
        .then((res) =>
            res.json().then((data) => {
                alert(data.status)
                setSyncLoading(false)
            })
        );
    }

    const clearLogs = () => {
        setSyncLoading(true)
        fetch("http://127.0.0.1:5000/clearLogs", {
            method: 'GET',
            headers: {
              'Accept': 'application/json',
              'Content-Type': 'application/json'
            }})
        .then((res) =>
            res.json().then((data) => {
                alert(data.status)
                setSyncLoading(false)
            })
        );
    }

  return (
    <div className='admin-main'>
        <h1>Admin View</h1>
        <div className='admin-query-div'>
            <label id="admin-query-label">Write query here: </label>
            <input id="admin-query-input"type="text" onChange={handleChange} value={query}/>
        </div>
        <div className='admin-query-controls'>
            <button id="admin-query-btn" onClick={() => queryLLM()}>Query</button>
            {syncLoading === false
            ? 
            <>
                <button id="admin-query-btn" onClick={() => {syncLogs();setQuery('');setAnswer({response: ''})}}>Sync</button>
                <button id="admin-clear-btn" onClick={() => clearLogs()}>Clear Logs</button>
            </>
            :
            <label><CircularProgress /></label>
            }
            <button id="admin-query-btn" onClick={() => {setQuery('');setAnswer({response: ''})}}>Reset</button>
            <a
                href={WorkerLogs}
                download={currentDate}
                target="_blank"
                rel="noreferrer"
            >
            <button onClick={() => setCurrentDate(new Date().toLocaleDateString())}>Download Worker Logs</button>
            </a>
        </div>
        <h3><label>{answer.response}</label></h3>
    </div>

  )
}

export default Admin