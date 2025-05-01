<template>
    <form @submit.prevent="onSubmit" class="prediction-form">
      <div class="grid-container">
        <!-- Player 1 Selection -->
        <div class="player-section player1-section">
          <h2>Player 1</h2>
          <div class="form-group">
            <label>Select Player:</label>
            <select v-model="selectedPlayer1" @change="handlePlayer1Change">
              <option v-for="player in players" :key="`p1-${player.name}`" :value="player.name">
                {{ player.name }}
              </option>
            </select>
          </div>
  
          <div v-if="showCustom1" class="custom-player-form">
            <div class="form-group">
              <label>Rank:</label>
              <input type="number" v-model.number="customPlayer1.rank">
            </div>
            <div class="form-group">
              <label>Seed:</label>
              <input type="number" v-model.number="customPlayer1.seed">
            </div>
            <div class="form-group">
              <label>Country (IOC):</label>
              <select v-model="customPlayer1.ioc">
                <option v-for="code in iocCodes" :key="`p1-ioc-${code}`" :value="code">{{ code }}</option>
              </select>
            </div>
            <div class="form-group">
              <label>Age:</label>
              <input type="number" v-model.number="customPlayer1.age" step="0.1">
            </div>
            <div class="form-group">
              <label>Height (cm):</label>
              <input type="number" v-model.number="customPlayer1.ht">
            </div>
          </div>
  
          <div v-else class="player-info">
            <p><span class="label">Rank:</span> {{ player1.rank }}</p>
            <p><span class="label">Seed:</span> {{ player1.seed }}</p>
            <p><span class="label">Country:</span> {{ player1.ioc }}</p>
            <p><span class="label">Age:</span> {{ player1.age }}</p>
            <p><span class="label">Height:</span> {{ player1.ht }} cm</p>
          </div>
        </div>
  
        <!-- Player 2 Selection -->
        <div class="player-section player2-section">
          <h2>Player 2</h2>
          <div class="form-group">
            <label>Select Player:</label>
            <select v-model="selectedPlayer2" @change="handlePlayer2Change">
              <option v-for="player in players" :key="`p2-${player.name}`" :value="player.name">
                {{ player.name }}
              </option>
            </select>
          </div>
  
          <div v-if="showCustom2" class="custom-player-form">
            <div class="form-group">
              <label>Rank:</label>
              <input type="number" v-model.number="customPlayer2.rank">
            </div>
            <div class="form-group">
              <label>Seed:</label>
              <input type="number" v-model.number="customPlayer2.seed">
            </div>
            <div class="form-group">
              <label>Country (IOC):</label>
              <select v-model="customPlayer2.ioc">
                <option v-for="code in iocCodes" :key="`p2-ioc-${code}`" :value="code">{{ code }}</option>
              </select>
            </div>
            <div class="form-group">
              <label>Age:</label>
              <input type="number" v-model.number="customPlayer2.age" step="0.1">
            </div>
            <div class="form-group">
              <label>Height (cm):</label>
              <input type="number" v-model.number="customPlayer2.ht">
            </div>
          </div>
  
          <div v-else class="player-info">
            <p><span class="label">Rank:</span> {{ player2.rank }}</p>
            <p><span class="label">Seed:</span> {{ player2.seed }}</p>
            <p><span class="label">Country:</span> {{ player2.ioc }}</p>
            <p><span class="label">Age:</span> {{ player2.age }}</p>
            <p><span class="label">Height:</span> {{ player2.ht }} cm</p>
          </div>
        </div>
      </div>
  
      <!-- Match Details -->
      <div class="match-details-section">
        <h2>Match Details</h2>
        <div class="form-group-row">
          <div class="form-group">
            <label>Surface:</label>
            <select v-model="surface">
              <option v-for="s in surfaces" :key="s" :value="s">{{ s }}</option>
            </select>
          </div>
          <div class="form-group">
            <label>Tournament Level:</label>
            <select v-model="tourneyLevel">
              <option v-for="level in tourneyLevels" :key="level.code" :value="level.code">
                {{ level.name }}
              </option>
            </select>
          </div>
        </div>
      </div>
  
      <!-- Submit Button -->
      <div class="submit-container">
        <button type="submit" :disabled="loading">
          {{ loading ? 'Predicting...' : 'Predict Winner' }}
        </button>
      </div>
    </form>
  </template>
  
  <script>
  export default {
    data() {
      return {
        selectedPlayer1: 'Novak Djokovic',
        selectedPlayer2: 'Carlos Alcaraz',
        player1: null,
        player2: null,
        showCustom1: false,
        showCustom2: false,
        customPlayer1: {
          rank: 100,
          seed: 0,
          ioc: 'USA',
          age: 25,
          ht: 185
        },
        customPlayer2: {
          rank: 100,
          seed: 0,
          ioc: 'USA',
          age: 25,
          ht: 185
        },
        surface: 'Hard',
        tourneyLevel: 'G',
        loading: false,
        
        // Constant data
        players: [
          { name: 'Novak Djokovic', rank: 1, seed: 1, ioc: 'SRB', age: 36, ht: 188 },
          { name: 'Carlos Alcaraz', rank: 2, seed: 2, ioc: 'ESP', age: 20, ht: 183 },
          { name: 'Daniil Medvedev', rank: 3, seed: 3, ioc: 'RUS', age: 27, ht: 198 },
          { name: 'Jannik Sinner', rank: 4, seed: 4, ioc: 'ITA', age: 22, ht: 188 },
          { name: 'Stefanos Tsitsipas', rank: 5, seed: 5, ioc: 'GRE', age: 25, ht: 193 },
          { name: 'Andrey Rublev', rank: 6, seed: 6, ioc: 'RUS', age: 26, ht: 188 },
          { name: 'Holger Rune', rank: 7, seed: 7, ioc: 'DEN', age: 20, ht: 188 },
          { name: 'Casper Ruud', rank: 8, seed: 8, ioc: 'NOR', age: 24, ht: 183 },
          { name: 'Taylor Fritz', rank: 9, seed: 9, ioc: 'USA', age: 25, ht: 193 },
          { name: 'Alexander Zverev', rank: 10, seed: 10, ioc: 'GER', age: 26, ht: 198 },
          { name: 'Frances Tiafoe', rank: 11, seed: 11, ioc: 'USA', age: 25, ht: 188 },
          { name: 'Tommy Paul', rank: 12, seed: 12, ioc: 'USA', age: 26, ht: 185 },
          { name: 'Felix Auger-Aliassime', rank: 13, seed: 13, ioc: 'CAN', age: 23, ht: 193 },
          { name: 'Ben Shelton', rank: 14, seed: 14, ioc: 'USA', age: 21, ht: 193 },
          { name: 'Karen Khachanov', rank: 15, seed: 15, ioc: 'RUS', age: 27, ht: 198 },
          { name: 'Hubert Hurkacz', rank: 16, seed: 16, ioc: 'POL', age: 26, ht: 196 },
          { name: 'Custom Player', rank: null, seed: null, ioc: '', age: null, ht: null }
        ],
        surfaces: ['Hard', 'Clay', 'Grass', 'Carpet'],
        tourneyLevels: [
          { code: 'G', name: 'Grand Slam' },
          { code: 'M', name: 'Masters 1000' },
          { code: 'A', name: 'ATP 500' },
          { code: 'D', name: 'ATP 250' },
          { code: 'F', name: 'Tour Finals' },
          { code: 'C', name: 'Challenger' }
        ],
        iocCodes: ['USA', 'ESP', 'SRB', 'RUS', 'GBR', 'FRA', 'GER', 'ITA', 'AUS', 'ARG', 'CAN', 'JPN', 'SUI', 'CZE', 'POL', 'BRA', 'NOR', 'AUT', 'BEL', 'CHI', 'GRE', 'DEN']
      };
    },
    created() {
      // Initialize player objects
      this.player1 = this.findPlayerByName(this.selectedPlayer1);
      this.player2 = this.findPlayerByName(this.selectedPlayer2);
    },
    methods: {
      findPlayerByName(name) {
        return this.players.find(p => p.name === name);
      },
      handlePlayer1Change() {
        this.player1 = this.findPlayerByName(this.selectedPlayer1);
        this.showCustom1 = this.selectedPlayer1 === 'Custom Player';
      },
      handlePlayer2Change() {
        this.player2 = this.findPlayerByName(this.selectedPlayer2);
        this.showCustom2 = this.selectedPlayer2 === 'Custom Player';
      },
      onSubmit() {
        this.loading = true;
        
        // Prepare data for API call
        const p1 = this.showCustom1 ? {
          player1_seed: this.customPlayer1.seed || 0,
          player1_rank: this.customPlayer1.rank || 100,
          player1_ioc: this.customPlayer1.ioc,
          player1_age: this.customPlayer1.age || 25,
          player1_ht: this.customPlayer1.ht || 185
        } : {
          player1_seed: this.player1.seed || 0,
          player1_rank: this.player1.rank || 100,
          player1_ioc: this.player1.ioc,
          player1_age: this.player1.age,
          player1_ht: this.player1.ht
        };
  
        const p2 = this.showCustom2 ? {
          player2_seed: this.customPlayer2.seed || 0,
          player2_rank: this.customPlayer2.rank || 100,
          player2_ioc: this.customPlayer2.ioc,
          player2_age: this.customPlayer2.age || 25,
          player2_ht: this.customPlayer2.ht || 185
        } : {
          player2_seed: this.player2.seed || 0,
          player2_rank: this.player2.rank || 100,
          player2_ioc: this.player2.ioc,
          player2_age: this.player2.age,
          player2_ht: this.player2.ht
        };
  
        const inputData = {
          ...p1,
          ...p2,
          surface: this.surface,
          tourney_level: this.tourneyLevel
        };
  
        // Emit the data to parent component to handle API call
        this.$emit('prediction', inputData);
        
        setTimeout(() => {
          this.loading = false;
        }, 500);
      }
    }
  };
  </script>
  
  <style scoped>
  .prediction-form {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    font-family: Arial, sans-serif;
  }
  
  .grid-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }
  
  .player-section {
    padding: 15px;
    border-radius: 8px;
  }
  
  .player1-section {
    background-color: #e6f2ff;
  }
  
  .player2-section {
    background-color: #ffebee;
  }
  
  h2 {
    margin-top: 0;
    margin-bottom: 15px;
    font-size: 1.5rem;
    color: #333;
  }
  
  .form-group {
    margin-bottom: 12px;
  }
  
  .form-group label {
    display: block;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 0.9rem;
  }
  
  .form-group input, 
  .form-group select {
    width: 100%;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 1rem;
  }
  
  .player-info {
    margin-top: 15px;
    background-color: rgba(255, 255, 255, 0.5);
    padding: 10px;
    border-radius: 4px;
  }
  
  .player-info p {
    margin: 5px 0;
  }
  
  .player-info .label {
    font-weight: bold;
  }
  
  .match-details-section {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
  }
  
  .form-group-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  
  .submit-container {
    text-align: center;
  }
  
  button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    transition: background-color 0.3s;
  }
  
  button:hover {
    background-color: #45a049;
  }
  
  button:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
  
  @media (max-width: 768px) {
    .grid-container, .form-group-row {
      grid-template-columns: 1fr;
    }
  }
  </style>