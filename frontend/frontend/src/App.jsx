import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [form, setForm] = useState({
    player1_rank: '',
    player2_rank: '',
    player1_seed: '',
    player2_seed: '',
    player1_age: '',
    player2_age: '',
    player1_ht: '',
    player2_ht: '',
    surface: '0',
    tourney_level: '0',
    player1_ioc: '0'
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const predict = async () => {
    const response = await axios.post('/predict', form);
    setResult(response.data.prediction === 1 ? 'Player 1 wins!' : 'Player 2 wins!');
  };

  return (
    <div style={{ padding: 20, maxWidth: 400 }}>
      <h1>Tennis Match Predictor</h1>
      {Object.keys(form).map((key) => (
        <div key={key}>
          <label>{key.replaceAll('_', ' ')}:</label>
          <input
            type="number"
            name={key}
            value={form[key]}
            onChange={handleChange}
            style={{ width: '100%', marginBottom: 10 }}
          />
        </div>
      ))}
      <button onClick={predict}>Predict Winner</button>
      {result && <h3>{result}</h3>}
    </div>
  );
};

export default App;
