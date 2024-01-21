import React from 'react'
import { useState } from 'react'
import "./Admin.css"

function Admin() {
    const [query, setQuery] = useState('')
    const [answer, setAnswer] = useState({
        response: ''
    })

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
                // Setting a data from api
                setAnswer({
                    response: data.response
                });
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
        <button id="admin-query-btn" onClick={() => queryLLM()}>Query</button>
        <h3><label>{answer.response}</label></h3>
    </div>

  )
}

export default Admin