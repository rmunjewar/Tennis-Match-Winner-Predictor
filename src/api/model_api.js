// src/api/model_api.js

/**
 * Send prediction request to the backend API
 * @param {Object} data - Player and match data for prediction
 * @returns {Promise<Object>} - Prediction result with winner and confidence
 */
export async function getPrediction(data) {
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      return result;
    } catch (error) {
      console.error("Error fetching prediction:", error);
      throw error; // Re-throw to be caught by the component
    }
  }
  
  /**
   * For development and testing without a backend
   * @param {Object} data - Player and match data
   * @returns {Promise<Object>} - Mock prediction result
   */
  export async function getMockPrediction(data) {
    return new Promise((resolve) => {
      // Simulate API delay
      setTimeout(() => {
        // Simple logic for demo purposes
        // In reality, this would be handled by the ML model
        const p1Rank = data.player1_rank || 100;
        const p2Rank = data.player2_rank || 100;
        
        // Player with better rank has higher chance of winning
        const p1Advantage = Math.max(0, (p2Rank - p1Rank) / 100);
        const baseProb = 0.5 + p1Advantage;
        
        // Add some randomness
        const randomFactor = Math.random() * 0.3 - 0.15;
        const p1WinProb = Math.min(Math.max(baseProb + randomFactor, 0.05), 0.95);
        
        const winner = Math.random() < p1WinProb ? 1 : 0;
        const confidence = winner === 1 ? p1WinProb : (1 - p1WinProb);
        
        resolve({
          winner: winner,
          confidence: confidence.toFixed(2)
        });
      }, 800); // Simulate network delay
    });
  }