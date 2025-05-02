<template>
  <div class="app-container">
    <div class="app-content">
      <header>
        <h1>Tennis Match Predictor</h1>
        <p class="subtitle">
          Predict the outcome using multiple machine learning models
        </p>
      </header>

      <PredictionForm @prediction="handlePrediction" :is-loading="isLoading" />

      <div v-if="isLoading" class="loading-message">
        <p>Generating predictions...</p>
      </div>

      <div
        v-if="allPredictions && !isLoading"
        class="prediction-results-container"
      >
        <h2>Prediction Results</h2>
        <div class="model-predictions">
          <div
            v-for="(prediction, modelName) in allPredictions"
            :key="modelName"
            class="prediction-result-card"
          >
            <h3>{{ formatModelName(modelName) }}</h3>
            <div v-if="!prediction.error" class="result-content">
              <div class="winner-announcement">
                Predicted Winner:
                <span
                  class="winner"
                  :class="{
                    player1: prediction.prediction === 1,
                    player2: prediction.prediction === 0,
                  }"
                >
                  Player {{ prediction.prediction === 1 ? "1" : "2" }}
                </span>
              </div>
              <div class="confidence-meter">
                <p>
                  P(Player 1 Wins):
                  {{ Math.round(prediction.probability.player1_wins * 100) }}%
                </p>
                <div class="progress-container">
                  <div
                    class="progress-bar player1-prob"
                    :style="{
                      width: `${prediction.probability.player1_wins * 100}%`,
                    }"
                  ></div>
                </div>
              </div>
            </div>
            <div v-else class="error-message small-error">
              Prediction failed: {{ prediction.error }}
            </div>
          </div>
        </div>
      </div>

      <div v-if="error && !isLoading" class="error-message">
        <p><strong>Error:</strong> {{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import PredictionForm from "./components/PredictionForm.vue";
// api stuff for Later
// Import your API service if you refactored it, otherwise use fetch directly
// import { getPrediction } from './apiService'; // Example if you have apiService.js

export default {
  components: {
    PredictionForm,
  },
  data() {
    return {
      allPredictions: null,
      error: null,
      isLoading: false,
    };
  },
  methods: {
    async handlePrediction(inputData) {
      this.isLoading = true;
      this.error = null;
      this.allPredictions = null;

      console.log("Sending data to backend:", inputData);

      try {
        const apiUrl = "http://localhost:5001/api/predict";

        const response = await fetch(apiUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(inputData),
        });

        const result = await response.json();

        console.log("Received response:", result);

        if (!response.ok) {
          throw new Error(
            result.error || `HTTP error! Status: ${response.status}`
          );
        }

        if (result.error) {
          throw new Error(result.error);
        }

        if (!result.predictions) {
          throw new Error(
            "Invalid response format from server: 'predictions' key missing."
          );
        }

        this.allPredictions = result.predictions;
      } catch (err) {
        console.error("Prediction failed:", err);
        this.error =
          err.message ||
          "Failed to get prediction. Please check the console for details.";
        this.allPredictions = null;
      } finally {
        this.isLoading = false;
      }
    },
    formatModelName(name) {
      switch (name) {
        case "random_forest":
          return "Random Forest";
        case "knn":
          return "K-Nearest Neighbors";
        case "decision_tree":
          return "Decision Tree";
        default:
          return name
            .replace(/_/g, " ")
            .replace(/\b\w/g, (l) => l.toUpperCase());
      }
    },

    getConfidenceInPrediction(prediction) {
      if (prediction.winner === 1) {
        return Math.round(prediction.confidence_player1_wins * 100);
      } else if (prediction.winner === 2) {
        return Math.round((1 - prediction.confidence_player1_wins) * 100);
      }
      return 0;
    },
  },
};
</script>

<style>
:root {
  --primary-color: #3498db;
  --secondary-color: #2ecc71;
  --player1-color: #3498db; /* blue for Player 1 for rn*/
  --player2-color: #e74c3c; /* red for Player 2 for rn*/
  --bg-light: #f9f9f9;
  --text-dark: #333;
  --error-color: #e74c3c;
  --border-color: #eee;
  --card-bg: #ffffff;
  --shadow-color: rgba(0, 0, 0, 0.1);
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
  line-height: 1.6;
  color: var(--text-dark);
  background-color: var(--bg-light);
}

.app-container {
  min-height: 100vh;
  padding: 20px;
}

.app-content {
  max-width: 1100px; /* Wider for multiple cards */
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

.loading-message {
  text-align: center;
  margin-top: 40px;
  font-size: 1.2rem;
  color: #555;
}

/* Container for all model prediction cards */
.prediction-results-container {
  margin-top: 30px;
  padding: 25px;
  background: var(--card-bg);
  border-radius: 10px;
  box-shadow: 0 5px 15px var(--shadow-color);
}

.prediction-results-container h2 {
  text-align: center;
  margin-bottom: 25px;
  font-size: 1.8rem;
  color: #2c3e50;
}

/* Grid layout for prediction cards */
.model-predictions {
  display: grid;
  /* Adjust minmax for desired card width */
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px; /* Spacing between cards */
}

/* Individual prediction card styling */
.prediction-result-card {
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 20px;
  background-color: var(--bg-light); /* Slightly different bg */
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  align-items: center; /* Center content */
}

.prediction-result-card h3 {
  text-align: center;
  margin-bottom: 15px;
  color: var(--primary-color);
  font-size: 1.4rem; /* Slightly larger */
  font-weight: 600;
}

.prediction-result-card .result-content {
  width: 100%; /* Ensure content takes full width */
}

.prediction-result-card .winner-announcement {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 15px;
  text-align: center;
}

/* Style the winner span */
.winner {
  font-weight: bold;
  padding: 2px 6px;
  border-radius: 4px;
  color: white; /* Text color for winner badges */
}
.winner.player1 {
  background-color: var(--player1-color);
}
.winner.player2 {
  background-color: var(--player2-color);
}

.prediction-result-card .confidence-meter {
  width: 100%;
  max-width: 300px; /* Max width for the meter itself */
  margin: 0 auto; /* Center the meter */
}

.prediction-result-card .confidence-meter p {
  font-size: 1rem;
  margin-bottom: 8px;
  text-align: center;
  color: #555;
}

.prediction-result-card .progress-container {
  height: 20px; /* Slightly smaller */
  background-color: #e0e0e0;
  border-radius: 10px;
  overflow: hidden;
  width: 100%; /* Make container full width */
}

.prediction-result-card .progress-bar {
  height: 100%;
  /* background: linear-gradient(to right, var(--player2-color), var(--player1-color)); */
  border-radius: 10px; /* Match container */
  transition: width 0.5s ease-out;
  text-align: center; /* For optional text inside */
  color: white;
  font-size: 0.8rem;
  line-height: 20px;
}
/* Specific bar for P(P1 Wins) */
.progress-bar.player1-prob {
  background-color: var(--player1-color); /* Solid color for P1 */
}

.error-message {
  margin-top: 20px;
  padding: 15px;
  background-color: rgba(231, 76, 60, 0.1);
  border-left: 4px solid var(--error-color);
  color: var(--error-color);
  border-radius: 4px;
}
.error-message.small-error {
  margin-top: 15px;
  padding: 10px;
  font-size: 0.9em;
  text-align: center;
  width: 100%;
}

@media (max-width: 768px) {
  h1 {
    font-size: 2rem;
  }
  .prediction-result-card h3 {
    font-size: 1.2rem;
  }
  .winner-announcement {
    font-size: 1rem;
  }
  .model-predictions {
    grid-template-columns: 1fr; /* Stack cards on smaller screens */
  }
}
</style>
