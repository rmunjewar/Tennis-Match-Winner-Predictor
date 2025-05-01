<template>
    <div class="app-container">
      <div class="app-content">
        <header>
          <h1>Tennis Match Predictor</h1>
          <p class="subtitle">Predict the outcome of tennis matches using machine learning</p>
        </header>
        
        <PredictionForm @prediction="handlePrediction" />
        
        <div v-if="predictionResult" class="prediction-result">
          <h2>Prediction Result</h2>
          <div class="result-content">
            <div class="winner-announcement">
              <span class="winner" :class="{ 'player1': predictionResult.winner === 1, 'player2': predictionResult.winner === 0 }">
                {{ predictionResult.winner === 1 ? 'Player 1' : 'Player 2' }} is predicted to win!
              </span>
            </div>
            
            <div class="confidence-meter">
              <p>Confidence: {{ Math.round(predictionResult.confidence * 100) }}%</p>
              <div class="progress-container">
                <div class="progress-bar" :style="{ width: `${predictionResult.confidence * 100}%` }"></div>
              </div>
            </div>
          </div>
        </div>
        
        <div v-if="error" class="error-message">
          <p>{{ error }}</p>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import PredictionForm from './components/PredictionForm.vue';
  
  export default {
    components: {
      PredictionForm
    },
    data() {
      return {
        predictionResult: null,
        error: null
      };
    },
    methods: {
      async handlePrediction(inputData) {
        try {
          // api stuff to deal with later
          
          /*
          const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(inputData)
          });
          
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          
          const result = await response.json();
          */
          
          const mockResult = {
            winner: Math.random() > 0.5 ? 1 : 0,
            confidence: (Math.random() * 0.5 + 0.5).toFixed(2)
          };
          
          this.predictionResult = mockResult;
          this.error = null;
        } catch (err) {
          this.error = err.message || 'Failed to get prediction.';
          this.predictionResult = null;
        }
      }
    }
  };
  </script>
  
  <style>
  :root {
    --primary-color: #3498db;
    --secondary-color: #2ecc71;
    --player1-color: #3498db;
    --player2-color: #e74c3c;
    --bg-light: #f9f9f9;
    --text-dark: #333;
    --error-color: #e74c3c;
  }
  
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-dark);
    background-color: var(--bg-light);
  }
  
  .app-container {
    min-height: 100vh;
    padding: 20px;
  }
  
  .app-content {
    max-width: 1000px;
    margin: 0 auto;
  }
  
  header {
    text-align: center;
    margin-bottom: 30px;
  }
  
  h1 {
    font-size: 2.5rem;
    color: var(--primary-color);
    margin-bottom: 10px;
  }
  
  .subtitle {
    font-size: 1.1rem;
    color: #666;
  }
  
  .prediction-result {
    margin-top: 30px;
    padding: 20px;
    background: linear-gradient(to right, rgba(52, 152, 219, 0.1), rgba(46, 204, 113, 0.1));
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  }
  
  .prediction-result h2 {
    text-align: center;
    margin-bottom: 20px;
    font-size: 1.8rem;
    color: #2c3e50;
  }
  
  .result-content {
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  
  .winner-announcement {
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 20px;
    text-align: center;
  }
  
  .winner.player1 {
    color: var(--player1-color);
  }
  
  .winner.player2 {
    color: var(--player2-color);
  }
  
  .confidence-meter {
    width: 100%;
    max-width: 400px;
  }
  
  .confidence-meter p {
    font-size: 1.1rem;
    margin-bottom: 10px;
    text-align: center;
  }
  
  .progress-container {
    height: 24px;
    background-color: #e0e0e0;
    border-radius: 12px;
    overflow: hidden;
  }
  
  .progress-bar {
    height: 100%;
    background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
    border-radius: 12px;
    transition: width 0.5s ease-out;
  }
  
  .error-message {
    margin-top: 20px;
    padding: 15px;
    background-color: rgba(231, 76, 60, 0.1);
    border-left: 4px solid var(--error-color);
    color: var(--error-color);
    border-radius: 4px;
  }
  
  @media (max-width: 768px) {
    h1 {
      font-size: 2rem;
    }
    
    .winner-announcement {
      font-size: 1.2rem;
    }
  }
  </style>