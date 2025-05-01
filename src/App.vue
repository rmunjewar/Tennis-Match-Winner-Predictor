<template>
    <div>
      <h1>Tennis Match Predictor</h1>
      <PredictionForm @prediction="handlePrediction" />
      <div v-if="predictionResult">
        <h2>Prediction:</h2>
        <p v-if="predictionResult.winner === 1">Player 1 is predicted to win!</p>
        <p v-else>Player 2 is predicted to win!</p>
        <p>Confidence: {{ predictionResult.confidence }}</p>
      </div>
      <div v-if="error" class="error">
        Error: {{ error }}
      </div>
    </div>
  </template>
  
  <script>
  import PredictionForm from './components/PredictionForm.vue';
  import { getPrediction } from './api/model_api.js';
  
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
          this.predictionResult = await getPrediction(inputData);
          this.error = null;
        } catch (err) {
          this.error = err.message || 'Failed to get prediction.';
          this.predictionResult = null;
        }
      }
    }
  };
  </script>
  
  <style scoped>
  .error {
    color: red;
  }
  </style>